# comfyui-anima-consistency

Nodes ComfyUI para consistência **sem treino** no ANIMA (Cosmos-Predict2 DiT). Instalação:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/adbrasi/comfyui-anima-consistency
```

Reinicie o ComfyUI. Os nodes aparecem em **anima/consistency**. Funciona com o Anima nativo do ComfyUI
(`Load Diffusion Model` → anima-base/aesthetic); não precisa de dependências extras.

## Fluxo recomendado (consistência de cena / paleta, sem máscara)

```
Load Diffusion Model ─► Anima Shared Attention (SPDA) ─► Anima Palette Lock ─► KSampler.model
Load CLIP ─► Anima Batch Prompts ─ positive ─► KSampler.positive
                                 ─ negative ─► KSampler.negative
                                 ─ latent   ─► KSampler.latent_image   (NÃO use Empty Latent Image)
KSampler ─► VAE Decode ─► Save Image   (sai uma imagem por linha de prompt)
```

- `prefix`: o que deve ser igual em todas (personagem, **ambiente**, estilo, iluminação).
- `prompts`: uma linha por imagem (ação/enquadramento).
- SPDA com `anchors = 1`: todas as imagens "olham" para a imagem 0 como um painel ao lado → mesmo ambiente/tema.
  `anchors = 0`: todas com todas (puxa para um visual médio do grupo).
- Palette Lock: alinha média/desvio por canal do latente previsto à imagem 0 nos passos iniciais → mesma paleta e tom.

## Nodes

### Anima Batch Prompts
Um prompt por imagem do batch: `prefix` (compartilhado) + uma linha por imagem.
Saídas: `positive`, `negative` (batch), `latent` (vazio, já com batch = nº de linhas, `width`×`height`) e `batch_size`.
Com batch 1 o ComfyUI descarta silenciosamente todos os prompts menos o primeiro e o SPDA fica desligado —
por isso o node entrega o latente pronto.

### Anima Palette Lock
Patch do MODEL (função pós-CFG). Enquanto σ ≥ `until_sigma`, puxa estatísticas globais por canal (média/desvio)
do x0 previsto de cada imagem para as da referência (`first image` ou `batch average`), com peso `strength`.
Afeta cor, tom e luz globais; não copia conteúdo nem layout. Ponto de partida: strength 0.6, until_sigma 0.5.

### Anima Shared Attention (SPDA)
Patch do MODEL. Cada imagem do batch também "olha" para as outras como se fossem **painéis ao lado**
(deslocamento RoPE de largura), numa softmax separada, misturada com peso λ, só nos blocos do meio.

| parâmetro | padrão recomendado | nota |
|---|---|---|
| strength | 0.4 | λ enquanto o layout se forma (σ > 0.6) |
| strength_late | 0.4 | λ depois (σ < 0.6); 0.7 deixa o visual mais uniforme mas reduz a estética (medido) |
| block_start / block_end | 4 / 16 | melhor no benchmark; 6/20 também ok. Blocos 20–27 lavam a imagem; todos os blocos → mosaico |
| sigma_start / sigma_end | 1.0 / 0.0 | |
| anchors | 0 | 0 = todos com todos |
| position | side | `same` copia pose/layout (colapso) |

**Modo âncora** (personagem fixo, N cenas novas): conecte `anchor_mask`.
1. Gere a âncora sozinha (prompt da linha 0 + seed S) e passe por um node de remoção de fundo → MASK.
2. `Anima Batch Prompts` com a linha 0 = prompt da âncora e as demais = cenas; KSampler com a **mesma seed S**.
3. Os alvos emprestam apenas tokens do personagem da âncora, numa softmax única (equivale a misturar
   com gate por correspondência). `anchor_bias` > 0 reforça a âncora.

Use sempre o mesmo comportamento no positivo e no negativo (o node já faz isso — não compartilhar no
ramo incondicional destrói a imagem).
