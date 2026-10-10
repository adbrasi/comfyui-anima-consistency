# StoryDiffusion para o Anima — pesquisa e proposta de método

Objetivo: dada uma lista de prompts, gerar N imagens que mantenham **o que deve ficar fixo** (personagem,
ambiente, paleta, tema, estilo) e variem **o que cada prompt pede** (ação, pose, câmera), sem treino e
sem máscara manual.

Status: documento teórico. Nada aqui foi validado em GPU ainda; a seção 6 é o plano de validação.

---

## 1. A arquitetura do Anima (o que importa para consistência)

Verificado no código do ComfyUI (`comfy/ldm/cosmos/predict2.py`, `comfy/ldm/anima/model.py`,
`comfy/model_detection.py`) e no cabeçalho do checkpoint `anima-turbo-v1.1.safetensors`.

| peça | detalhe | consequência |
|---|---|---|
| Denoiser | Cosmos-Predict2 `MiniTrainDIT`, ~2.0B params, **28 blocos**, largura 2048, **16 cabeças × 128**, MLP 8192 | um ponto de intervenção por bloco: o `self_attn` |
| Bloco | adaLN-LoRA (timestep) → **self-attn** → **cross-attn (só texto)** → MLP | ≠ MMDiT (FLUX/SD3): texto **não** está na self-attn |
| Posição | RoPE 3D *split-half* sobre (T,H,W): 22/21/21 pares de frequência por cabeça; extrapolação NTK h/w = 4.0 | dá para "mover" uma imagem no espaço girando as chaves (o que o SPDA já faz) |
| Eixo T | existe (modelo de vídeo na origem), imagens usam T = 1 | hipótese exótica: frames no eixo T (seção 4.7) |
| Patch / latente | VAE Wan2.1/Qwen-Image 16 canais ×8, patch 2 → 1024² = **64×64 = 4096 tokens** | custo da atenção compartilhada ~ N·4096 tokens |
| Texto | Qwen3-0.6B → **LLM adapter** (6 blocos, cross-attn a partir de embeddings de tokens **T5**) → até 512 tokens de 1024-d | a saída do adapter é **alinhada posição-a-posição com os tokens T5 do prompt** |
| Sampling | flow (shift 3), CFG ~4–5 (base) ou 1 (turbo, 8–12 passos) | com CFG 1 só existe ramo condicional |

Duas propriedades tornam o Anima **mais fácil** que FLUX para isso:

1. **Self-attn só de imagem.** Em MMDiT, compartilhar K/V também compartilha tokens de texto e contamina o
   prompt (AlignedGen precisou de *Selective Attention Sharing* para evitar isso). No Anima o texto entra
   por uma cross-attn separada: compartilhar a self-attn já é "só imagem" por construção.
2. **Embeddings de texto posicionais por token T5.** Os primeiros tokens de cada prompt são o `prefix`
   compartilhado, então dá para sincronizar exatamente a parte do texto que descreve o que é fixo
   (seção 4.5).

E uma propriedade que **atrapalha**: RoPE 3D. Se uma imagem vê as chaves da outra nas mesmas posições,
ocorre a "Positional Collision" descrita pelo AlignedGen: o modelo lê as duas como sobrepostas e copia o
layout (o `position = same` do SPDA colapsa por isso).

---

## 2. Trabalhos próximos — o mecanismo de cada um

| trabalho | base | mecanismo central | o que aproveitar no Anima |
|---|---|---|---|
| **StoryDiffusion** — Consistent Self-Attention (2024) | SD1.5/SDXL U-Net | na self-attn, cada imagem concatena às suas K/V **tokens amostrados aleatoriamente das outras imagens do batch** (taxa padrão 0.5; 0.3 falhou na ablação), **uma única softmax**, query intacta; janelas deslizantes para histórias longas | o esqueleto do método: batch + K/V compartilhadas, amostragem de tokens |
| **ConsiStory** (2024) | SDXL | atenção compartilhada **restrita ao sujeito** (máscara via cross-attn), *feature injection* por correspondência, *query blending/dropout* para preservar variedade de layout | ideia de dosar a parte "explorar" (layout) separada da parte "seguir" (identidade). A máscara não serve para você: o alvo é a cena inteira |
| **StyleAligned** (2024) | SDXL | cada imagem atende à **imagem de referência** com **AdaIN** nas Q/K para alinhar estatísticas | AdaIN em Q/K → alinhamento de estilo/paleta |
| **AlignedGen** (2025) | FLUX.1-dev | **ShiftPE**: referência colocada virtualmente **ao lado** (índices `(i,j) → (i,j+w)`), **key scaling** λ≈1.1 nas chaves da referência, só em parte das camadas | **valida o "side" do SPDA**; key scaling como controle de força numa softmax única |
| **Story2Board** (2025) | FLUX.1-dev | cada item do batch é um canvas de 2 painéis [referência / cena]; **Latent Panel Anchoring**: metade-referência sobrescrita pela do item 1 após cada bloco; **RAVM**: pares de tokens com atenção recíproca alta (min das duas direções, EMA, Otsu) misturam **só os values** (λ=0.5), preservando layout | RAVM é o melhor candidato para detalhe fino sem copiar pose |
| **CharaConsist** (ICCV 2025) | FLUX.1 | *point-tracking attention* (casa posições entre imagens e re-codifica a posição), *adaptive token merge* ponderado pela similaridade do match e decaindo no tempo; separa frente/fundo para cena fixa vs. variável | re-posicionar as chaves por correspondência = versão "inteligente" do ShiftPE |
| **1Prompt1Story** (ICLR 2025) | SDXL | todos os prompts num prompt só (*context consistency* do modelo de linguagem) + reponderação por valores singulares + cross-attn que preserva identidade | sincronizar a parte de identidade do embedding de texto |
| **Infinite-Story** (2025) | AR por escala | *Identity Prompt Replacement* (combate o viés de contexto do encoder de texto), *Adaptive Style Injection*, *Synchronized Guidance Adaptation* | mesmo diagnóstico: o encoder "puxa" a identidade conforme o resto do prompt |

Lição comum dos trabalhos em DiT (AlignedGen, CharaConsist): **compartilhar atenção ingenuamente em DiT falha
por causa da posição**. A posição dos tokens emprestados é o parâmetro de projeto mais importante.

---

## 3. Por que o método atual (SPDA) não basta

O que já existe no repo: blend de duas softmax `own + λ·(borrowed − own)`, chaves emprestadas deslocadas
`W·(j−i)` em largura, blocos 4–16.

Problemas:

1. **Um escalar λ cobre tudo** (identidade, cenário, paleta, composição). Subir λ uniformiza o que deveria
   variar (pose, câmera); baixar perde o que deveria ficar.
2. **Deslocamento cresce com N.** Com 6 imagens a 1024², a imagem 0 vê a 5 a 320 tokens de distância.
   Relações tão longas quase não aparecem no treino, então a atenção a painéis distantes enfraquece ou
   fica aleatória, e a consistência piora conforme a história cresce.
3. **Custo O(N²)** (cada imagem atende a todas): histórias longas não cabem.
4. **Nada sincroniza o texto**: o mesmo `prefix` vira embeddings diferentes em cada frame, porque o adapter
   e o Qwen veem o resto do prompt.

---

## 4. Proposta: *Anima Story* — um controle por aspecto

O que fica fixo, separado por tipo, com um mecanismo (e um botão) para cada:

| o que fica fixo | mecanismo | botão |
|---|---|---|
| personagem + ambiente (conteúdo) | 4.1 Story Attention (softmax única, tokens amostrados, posição em anel) | `share` |
| detalhes finos (roupa, objetos) sem copiar pose | 4.2 Reciprocal Value Mixing | `detail` |
| paleta / tom / luz | 4.3 Palette Lock (já implementado) | `palette` |
| composição global (opcional) | 4.4 ruído inicial parcialmente compartilhado | `composition` |
| descrição textual do que é fixo | 4.5 sincronização do prefix | `text_sync` |

O que varia (pose, ação, câmera) fica com cada prompt. O padrão de cada botão deixa isso livre.

### 4.1 Story Attention (núcleo, = StoryDiffusion adaptado a DiT)

Nos blocos selecionados, para a imagem *i*:

```
K_i' = [ K_i ; shift(K_src, pos) ]       V_i' = [ V_i ; V_src ]
logits dos tokens emprestados += b        (b = log-bias; AlignedGen usa key scaling ≈ equivalente)
out_i = softmax(Q_i K_i'^T / √d) V_i'     (uma única softmax, como StoryDiffusion)
```

- **Fontes:** `all` (todas as outras) ou `anchors` (as K primeiras). O modo âncora custa O(N·K), não O(N²).
- **Amostragem de tokens** (StoryDiffusion): pegar uma fração `r` (0.5) dos tokens das fontes, sorteada por
  passo. Reduz memória, e a parte não emprestada protege a liberdade de layout.
- **Posição em anel** (corrige o problema 2): em vez de `W·(j−i)`, colocar cada fonte num vizinho imediato
  (esquerda, direita, cima, baixo; depois as diagonais) usando os eixos W **e H** do RoPE. Toda fonte fica
  a ≤ 1 painel de distância, a mesma escala que o modelo vê em imagens largas, "multiple views" e
  "comic". Acima de 8 fontes, amostrar fontes por passo.
- **Softmax única vs. blend:** a softmax única deixa o modelo decidir por token quanto emprestar (tokens do
  personagem casam com o personagem, tokens do céu casam com o céu). Isso é um "gate por correspondência"
  natural, que é o que a máscara do ConsiStory fazia à mão, só que para a **cena inteira**. O blend
  (SPDA atual) fica como alternativa para comparação.
- **Agenda:** blocos do meio (ponto de partida 4–16, vindo do benchmark do lab); força maior nos σ altos,
  onde layout e cena se decidem, e menor nos baixos.

### 4.2 Reciprocal Value Mixing (detalhe fino, do Story2Board)

Num subconjunto de blocos tardios, depois da atenção:

1. Calcular a afinidade entre a imagem *i* e a fonte *a*, em média sobre as cabeças: `A_ia = softmax(Q_i K_a^T)`
   e `A_ai`.
2. Reciprocidade: `M = min(A_ia, A_aiᵀ)`, com EMA entre passos e limiar automático (Otsu).
3. Para os tokens de *i* com par confiável `u*`: `V_i[v] ← λ V_i[v] + (1−λ) V_a[u*]`.

Q e K não mudam, então **o layout da imagem i fica intacto**. Só a aparência dos pontos que já
correspondem é puxada. Isso ataca a falha clássica do compartilhamento de atenção (copiar a pose) por
construção, em vez de por ajuste de λ.

Custo: uma matriz 4096×4096 por par e por bloco (em média sobre as cabeças) é viável em GPU para poucos
blocos.

### 4.3 Palette Lock (já no repo: `Anima Palette Lock`)

Função pós-CFG: nos passos iniciais (σ ≥ `until_sigma`), alinha média e desvio **por canal** do x0 previsto
de cada imagem aos da referência. Fixa a paleta e o tom globais sem tocar no conteúdo.

Variante a testar: AdaIN nas Q/K da self-attn (StyleAligned), que atua em features em vez de no latente.

### 4.4 Ruído parcialmente compartilhado (opcional)

`ε_i = √(1−ρ)·ε_i + √ρ·ε_shared`. Correlaciona a composição de baixa frequência (distribuição de massas e
cores). Implementável como wrapper `OUTER_SAMPLE` no MODEL, sem trocar o KSampler.

Risco conhecido: ρ alto puxa a mesma pose. ρ entre 0.1 e 0.3 é o ponto de partida, e o padrão é 0.

### 4.5 Sincronização do prefix (específico do Anima)

O adapter devolve um vetor por token T5, na ordem do prompt. Então:

1. Codificar `prefix + linha_i` para cada i (como hoje).
2. Contar quantos tokens T5 o `prefix` ocupa (tokenizar o prefix sozinho e checar se a fronteira bate).
3. Substituir essas posições no embedding de cada frame pela **média** (ou pelas do frame 0), com mistura
   `text_sync ∈ [0,1]`.

É a ideia do Identity Prompt Replacement do Infinite-Story aplicada à saída do LLM adapter: o personagem
e o ambiente passam a ser descritos **pelos mesmos vetores** em todas as imagens. Custo zero na
amostragem.

### 4.6 Histórias longas

- **Modo âncora com re-geração:** um `anchor prompt` separado (ou a linha 0) entra como item 0 de **cada**
  lote, com a mesma seed. Como é determinístico, cada lote recria as mesmas K/V da âncora, e ela é
  descartada da saída. Assim dá para gerar 30 imagens em lotes de 4 com a mesma referência.
- Alternativa: cache de K/V da âncora por (bloco, passo), amostrado com r = 0.25 para caber na memória.

### 4.7 Hipóteses mais especulativas

- **Painel em canvas (Story2Board):** gerar cada frame como canvas largo [âncora | cena] com Latent Panel
  Anchoring. É o prior "comic / multiple views" nativo do modelo, mas custa 2× tokens e reduz a resolução
  útil. Serve como baseline forte de comparação.
- **Frames no eixo T:** concatenar os frames como tempo (T = N) numa única sequência. O Anima é derivado do
  Cosmos-Predict2-2B-*Text2Image*, provavelmente treinado só com T = 1. A rotação temporal seria fora da
  distribuição, mas é o "deslocamento sem colisão" mais barato possível. Vale um teste rápido.
- **Correspondência para posição (CharaConsist):** re-posicionar cada chave emprestada para a posição do
  token correspondente na imagem alvo, em vez de um deslocamento fixo.

---

## 5. Implementação no ComfyUI (como encaixa)

- Tudo como patch de MODEL:
  - `optimized_attention_override` para 4.1 e 4.2 (q,k,v chegam já com RoPE, em `(B,H,S,D)`; o
    `block_index` está em `transformer_options`);
  - wrapper `DIFFUSION_MODEL` para σ e grade;
  - `post_cfg_function` para 4.3;
  - wrapper `OUTER_SAMPLE` para 4.4.
- 4.5 entra no `Anima Batch Prompts`.
- Cuidados já resolvidos:
  - batch do latente = nº de prompts (o node entrega o latente);
  - positivo e negativo no mesmo dispositivo;
  - aplicar o mesmo patch nos ramos cond e uncond (não compartilhar no uncond destrói a imagem, medido
    no lab).
- Precisão: calcular as softmax compartilhadas em bf16 com SDPA; a afinidade de 4.2 em fp32 sobre a média
  das cabeças.

---

## 6. Plano de validação (para quando houver GPU)

**Conjunto de teste:** ~20 histórias × 4–6 prompts, com dois tipos:
- (a) mesmo personagem em cenas variadas;
- (b) mesmo **ambiente/paleta** com personagens ou ações variadas (o seu caso).

Seeds fixas.

**Métricas:**

| eixo | métrica |
|---|---|
| consistência | DreamSim / DINOv2 entre pares de imagens da história; distância de paleta (EMD de histograma em Lab) |
| obediência ao prompt | CLIP-T por imagem com o prompt da sua linha |
| diversidade de layout | variância de pose (keypoints) e de posição do personagem; "Scene Diversity" do Story2Board |
| qualidade | aesthetic score; inspeção visual |

**Ablação (uma coisa por vez sobre o baseline):**
1. base (sem nada);
2. SPDA atual;
3. 4.1 com posição linear vs. em anel;
4. 4.1 softmax única vs. blend;
5. \+ r = 0.5 vs. 1.0;
6. \+ 4.5 text_sync;
7. \+ 4.3 palette;
8. \+ 4.2 RAVM;
9. \+ 4.4 ruído.

**Critério de sucesso:** subir consistência **sem** perder CLIP-T nem diversidade de layout. Se uma mudança
só troca uma coisa pela outra, ela é um ajuste de parâmetro, não um mecanismo novo.

---

## 7. Ordem sugerida de implementação

1. 4.5 (text_sync): barato, isolado, sem risco para a imagem.
2. 4.1 com posição em anel + softmax única + amostragem de tokens. É o StoryDiffusion de verdade para o Anima.
3. 4.6 modo âncora com re-geração (histórias longas).
4. 4.2 RAVM, se 4.1 + 4.5 não segurarem detalhes finos.
5. 4.4 e 4.7 como experimentos.

## Fontes

- StoryDiffusion — https://arxiv.org/abs/2405.01434
- ConsiStory — https://dl.acm.org/doi/10.1145/3658157
- AlignedGen — https://arxiv.org/abs/2509.17088
- Story2Board — https://arxiv.org/abs/2508.09983
- CharaConsist — https://arxiv.org/abs/2507.11533
- 1Prompt1Story — https://arxiv.org/abs/2501.13554
- Infinite-Story — https://arxiv.org/abs/2511.13002
- Anima (model card) — https://huggingface.co/circlestone-labs/Anima ; Diffusers docs — https://huggingface.co/docs/diffusers/en/api/pipelines/anima
