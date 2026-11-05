import os
from functools import lru_cache

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

KNOWLEDGE_PATH = os.path.join("knowledge", "tumor_facts.txt")

DEFAULT_FACTS = """Glioma: Gliomas arise from glial cells, the supporting cells of the brain and spinal cord. They range from slow-growing low-grade tumors to aggressive high-grade tumors such as glioblastoma, and usually appear within the brain tissue itself.
Meningioma: Meningiomas arise from the meninges, the membranes that cover the brain and spinal cord. Most are benign and slow-growing, and they typically appear as well-defined masses attached to the dura, outside the brain tissue.
Pituitary Tumor: Pituitary tumors develop in the pituitary gland at the base of the brain. Most are benign adenomas. Some produce excess hormones, and larger ones can press on the optic nerves and affect vision.
No Tumor: The classifier found no pattern matching glioma, meningioma or pituitary tumor in this slice. This does not rule out other conditions, and a single 2D slice cannot represent the whole brain.
Model Limitations: This classifier works on single 2D MRI slices and four classes. Scans from other scanners, sequences or views may be out of distribution and can give unreliable or overconfident predictions.
"""


@lru_cache(maxsize=1)
def _get_vector_db():
    if not os.path.exists(KNOWLEDGE_PATH) or os.path.getsize(KNOWLEDGE_PATH) == 0:
        os.makedirs("knowledge", exist_ok=True)
        with open(KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
            f.write(DEFAULT_FACTS)

    documents = TextLoader(KNOWLEDGE_PATH, encoding="utf-8").load()
    splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=0)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)


def query_knowledge_base(query: str) -> str:
    try:
        results = _get_vector_db().similarity_search(query, k=2)
        if not results:
            return "No specific reference documents matched the query."
        return "\n\n".join(doc.page_content for doc in results)
    except Exception:
        return "No reference material was available for this query."
