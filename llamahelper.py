import requests
import numpy as np
import faiss

class LlamaHelper:
	def __init__(self, server_url="http://127.0.0.1:8989", max_tokens=1000):
		self.server_url = server_url
		self.max_tokens = max_tokens
		
	def get_embedding(self, text):
		response = requests.post(
			f"{self.server_url}/embedding",
			json={"content": text}
		)
		if response.status_code == 200:
			return np.array(response.json()[0]['embedding'][0])
		else:
			raise Exception(f"Error from llama.cpp: {response.text}")

	def generate_text(self, prompt):
		response = requests.post(
			f"{self.server_url}/v1/completions",
			json={"prompt": prompt, "max_tokens": self.max_tokens}
		)
		if response.status_code == 200:
			return response.json()["choices"][0]['text']
		else:
			raise Exception(f"Error from llama.cpp: {response.text}")

	def load_documents(self):
		with open("data.txt", "r") as f:
			text = f.read()
		chunks = text.split("\n\n")  # Simple split by paragraphs
		embeddings = np.array([self.get_embedding(chunk) for chunk in chunks])
		dimension = embeddings.shape[1]
		index = faiss.IndexFlatL2(dimension)
		index.add(embeddings)
		return chunks, index

	def search_relevant_chunks(self, query, k=5):
		query_embedding = self.get_embedding(query).reshape(1, -1)
		distances, indices = self.index.search(query_embedding, k)
		return [self.chunks[i] for i in indices[0]]

	def generate_response(self, query, relevant_chunks):
		context = "\n\n".join(relevant_chunks)
		prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
		print(prompt)
		return self.generate_text(prompt)
