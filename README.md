# comfyui-anima-consistency

Nodes ComfyUI para consistência **sem treino** no ANIMA (Cosmos-Predict2 DiT). Instalação:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/adbrasi/comfyui-anima-consistency
```

Reinicie o ComfyUI. Os nodes aparecem em **anima/consistency**. Funciona com o Anima nativo do ComfyUI
(`Load Diffusion Model` → anima-base/aesthetic); não precisa de dependências extras.

## Nodes

### Anima Batch Prompts
Um prompt por imagem do batch: `prefix` (personagem + qualidade, compartilhado) + uma linha por imagem.
Saídas: `conditioning` (batch) e `batch_size` → ligue em `Empty Latent Image (batch_size)`.

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
