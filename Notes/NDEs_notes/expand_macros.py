import re, pathlib

file = pathlib.Path("Kidger_on_NDEs.md")
text = file.read_text()

# Add all your macro expansions here:
replacements = {
    r'\\mR': r'\\mathbb{R}',
    r'\\dydt': r'\\frac{dy}{dt}',
}

for k, v in replacements.items():
    text = re.sub(k, v, text)

pathlib.Path(file.stem + "_github.md").write_text(text)
print("✅ Macros expanded →", file.stem + "_github.md")