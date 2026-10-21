import unittest
import os
from src.insight_agent import build_index, get_chroma_client

class TestInsightAgent(unittest.TestCase):
    
    def setUp(self):
        build_index()
        self.client = get_chroma_client()
        self.collection = self.client.get_collection(name="hackforge_knowledge")

    def test_rag_retrieval_lahore(self):
        results = self.collection.query(
            query_texts=["Is the air in Lahore safe for a morning walk?"],
            n_results=1
        )
        self.assertTrue(len(results['documents'][0]) > 0)
        self.assertIn("Lahore", results['documents'][0][0])
        self.assertEqual(results['metadatas'][0][0]['source'], "openaq_lahore.json")

    def test_rag_retrieval_mars(self):
        results = self.collection.query(
            query_texts=["What is the capital of Mars?"],
            n_results=1
        )
        self.assertTrue(len(results['documents'][0]) > 0)
        self.assertIn("Mars", results['documents'][0][0])

if __name__ == "__main__":
    unittest.main()