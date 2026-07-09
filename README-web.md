# Plot Lab — interface web (PyWebView)

Front em **HTML/CSS/JS** numa janela nativa; backend em **Python + matplotlib/seaborn**
(o que garante o export TIFF de publicação). O front manda a configuração, o Python
renderiza a figura e devolve o preview; o export grava o arquivo real em 300 dpi.

```
graph/
├── main_web.py            # ponto de entrada da versão web  ← rode este
├── api.py                 # ponte JS↔Python (reusa core/ inteiro)
├── web/index.html         # toda a interface (HTML+CSS+JS)
├── core/                  # núcleo intacto: data_loader, plot_registry, plot_engine, stats
├── app.py / ui_main.py    # versão antiga em CustomTkinter (continua funcionando)
├── requirements-web.txt
└── exemplo.csv
```

## Rodar (com Python)

```bash
pip install -r requirements-web.txt
python main_web.py
```

- **Windows**: usa o **Edge WebView2**, que já vem no Windows 10/11. Nada a instalar.
- **macOS**: usa o WebKit do sistema. Nada a instalar.
- **Linux**: precisa do WebKitGTK — `sudo apt install python3-gi gir1.2-webkit2-4.1`
  (ou `pip install "pywebview[qt]"`).

## Empacotar num executável

> **Não existe um único executável para todos os SOs.** PyInstaller gera um binário
> **para o sistema onde você roda o build**: faça um build no Windows para ter o `.exe`,
> outro no macOS para o `.app`, outro no Linux. O usuário final não precisa de Python.

A diferença crucial do empacotamento é incluir a pasta `web/` como dado e coletar o `webview`.

### Windows (gera PlotLab.exe)
```bat
pip install -r requirements-web.txt pyinstaller
pyinstaller --noconfirm --windowed --name PlotLab ^
  --add-data "web;web" ^
  --collect-all webview ^
  --collect-submodules sklearn ^
  main_web.py
```
Resultado em `dist\PlotLab\`. Zipe a pasta e envie — o usuário abre `PlotLab.exe`.
O destinatário só precisa do **WebView2 Runtime** (já presente em Win10/11; em máquinas
antigas, o instalador gratuito "Evergreen WebView2" da Microsoft resolve).

### macOS (gera PlotLab.app)
```bash
pip install -r requirements-web.txt pyinstaller
pyinstaller --noconfirm --windowed --name PlotLab \
  --add-data "web:web" \
  --collect-all webview \
  --collect-submodules sklearn \
  main_web.py
```
(No macOS/Linux o separador do `--add-data` é `:` em vez de `;`.)

### Linux
Mesmo comando do macOS. Garanta o WebKitGTK instalado na máquina de build
(`gir1.2-webkit2-4.1`). Saída em `dist/PlotLab/`.

### Observações
- Prefira o modo pasta (acima) ao `--onefile`: inicia mais rápido e gera menos
  alarme falso de antivírus. Para arquivo único, troque por `--onefile`.
- **UMAP**: se não for usar, deixe `umap-learn` desinstalado — ele puxa numba/llvmlite
  e complica/incha muito a build. PCA e t-SNE seguem funcionando sem ele.
- Se o app abrir em branco no Windows empacotado, quase sempre é o WebView2 Runtime
  ausente na máquina do usuário — instale o Evergreen WebView2.
