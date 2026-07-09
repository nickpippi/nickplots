# Plot Lab

App desktop em Python para análise de dados, estatística e figuras de publicação.
Motor: **matplotlib + seaborn** (exporta TIFF/PNG/PDF/SVG a 300 dpi). UI: **CustomTkinter**.

## Estrutura

```
plot_lab/
├── app.py                 # ponto de entrada
├── ui_main.py             # janela (CustomTkinter) — monta os controles a partir do schema
├── exemplo.csv            # dados de teste
├── requirements.txt
└── core/                  # núcleo, 100% sem dependência de UI
    ├── data_loader.py     # CSV, inferência de tipo, filtro query(), export CSV
    ├── plot_registry.py   # ★ schema declarativo dos gráficos (adicione gráficos aqui)
    ├── plot_engine.py     # validação, estilo/fundo, camadas, export
    └── stats.py           # PCA / t-SNE / UMAP (em thread), t-test / ANOVA
```

**Para adicionar um gráfico novo:** escreva uma função `render(ax, df, mapping, params)`
e registre um `PlotSpec` em `core/plot_registry.py`. A UI passa a oferecê-lo
automaticamente, com painel de configuração e validação de tipo. Não se mexe na UI.

## Rodar (você, que tem Python)

```bash
pip install -r requirements.txt
python app.py
```

Carregue `exemplo.csv` para testar. Em scatter/volcano, **clique num ponto** para ver
a linha correspondente. UMAP só funciona se instalar `umap-learn` (está comentado no
requirements porque pesa).

## Distribuir para quem NÃO tem Python

O destinatário não instala nada — você gera um executável e manda a pasta.
PyInstaller produz um binário **para o sistema onde você roda o build** (Windows gera
.exe; Mac gera .app; Linux gera ELF). Não dá para gerar .exe a partir do Mac/Linux.

### Windows (gerar o .exe)
Numa máquina Windows com Python:
```bash
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --windowed --name PlotLab ^
  --collect-all customtkinter ^
  --collect-submodules sklearn ^
  app.py
```
Saída em `dist/PlotLab/`. Zipe essa pasta inteira e envie. O usuário abre
`PlotLab/PlotLab.exe` com duplo-clique.

- `--windowed` evita abrir um terminal preto junto.
- `--collect-all customtkinter` é obrigatório (o PyInstaller não acha os temas dele sozinho).
- Se NÃO for usar UMAP, mantenha `umap-learn` desinstalado: ele puxa numba/llvmlite e
  complica/incha muito a build. Sem UMAP, PCA e t-SNE seguem funcionando.
- Prefira o modo pasta (acima) ao `--onefile`: inicia mais rápido e dá menos falso
  positivo de antivírus. Se quiser um arquivo único, troque por `--onefile`.

### macOS / Linux
Mesmo comando (troque `^` por `\` para quebra de linha). No Mac sai `dist/PlotLab.app`.

### Alternativa sem build (se a pessoa topar 1 instalação)
1. Instalar Python em python.org (marcar “Add Python to PATH” no Windows).
2. Na pasta do projeto: `pip install -r requirements.txt`
3. `python app.py`


## Novidades desta versão
- **Temas claro/escuro coerentes**: ao trocar para um tema escuro, fundo, texto e grade
  mudam juntos (antes o fundo branco fixo apagava o tema). Os color pickers viram
  *override opcional* por cima do tema; "Limpar overrides" volta ao tema.
- **Legenda controlável** (aba Legenda): mostrar/ocultar, posição (incl. "Fora à direita"),
  tamanho da fonte, nº de colunas, moldura. No modo **"Livre"** você **arrasta a legenda
  com o mouse** dentro do gráfico — a posição é salva (sliders X/Y acompanham).
- **Preview ao vivo**: tema, legenda, rótulos e parâmetros redesenham na hora.
- **Figura acompanha o preview**: ao redimensionar a janela, o gráfico se reajusta ao
  tamanho do painel — sem legenda gigante no meio da tela.
- **Interface em abas** (Gráfico / Estilo / Legenda / Análise) + alternância Dark/Light do app.
