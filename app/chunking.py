from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Initialize the text chunker.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                ", ",
                " ",
                ""
            ],
        )

    def chunk_text(self, text: str):
        """
        Chunk a string of text into smaller segments.
        Returns a list of strings.
        """
        return self.splitter.split_text(text)

    def chunk_documents(self, docs: list):
        """
        Chunk a list of LangChain Document objects.
        Returns a new list of Document chunks.
        """
        return self.splitter.split_documents(docs)
