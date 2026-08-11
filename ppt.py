from docling.document_converter import DocumentConverter

source = "/home/mahtwog/Downloads/BATCH 9A.pptx"  # a document via a local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"
