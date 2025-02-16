import json
import requests
import numpy as np
from chroma import ChromaDBClient
import os

class LlamaHelper:
	def __init__(self, server_url="http://127.0.0.1:8989", max_tokens=1000):
		self.server_url = server_url
		self.max_tokens = max_tokens
		self.chroma_client = ChromaDBClient(
			collection_name="documents",
			persist_directory="chroma_db"
		)
		self.chunks = []

	def get_embedding(self, text):
		response = requests.post(
			f"{self.server_url}/embedding",
			json={"content": text}
		)
		if response.status_code == 200:
			return response.json()[0]['embedding'][0]
		raise Exception(f"Error from llama.cpp: {response.text}")

	def generate_text(self, prompt):
		response = requests.post(
			f"{self.server_url}/v1/completions",
			json={"prompt": prompt, "max_tokens": self.max_tokens}
		)
		if response.status_code == 200:
			return response.json()["choices"][0]['text']
		raise Exception(f"Error from llama.cpp: {response.text}")

	def process_and_store_documents(self, file_path):
		with open(file_path, "r") as f:
			text = f.read()

		chunks = self._split_text(text)
		# embeddings = [self.get_embedding(chunk) for chunk in chunks]
		ids = [f"doc_{i}" for i in range(len(chunks))]

		self.chroma_client.add_documents(
			documents=chunks,
			metadatas=[{"source": file_path} for _ in chunks],
			ids=ids
		)
		self.chunks.extend(chunks)

	@staticmethod
	def _split_text(text, chunk_size=500):
		chunks = []

		# TODO
		# 1. split by paragraph
		# 2. break down into sentences
		# 3. if sentence bigger than chunk_size - split by chunk size

		paragraphs = text.split("\n\n")  # how about \r\n ?
		for paragraph in paragraphs:
			sentences = paragraph.split(". ")  # how to determine sentences correctly?
			for sentence in sentences:
				words = sentence.split()
				current_chunk = []
				current_length = 0

				for word in words:
					if current_length + len(word) + 1 > chunk_size:
						chunks.append(" ".join(current_chunk))
						current_chunk = []
						current_length = 0
					current_chunk.append(word)
					current_length += len(word) + 1

				if current_chunk:
					chunks.append(" ".join(current_chunk))

		# words = text.split()
		# chunks = []
		# current_chunk = []
		# current_length = 0
		#
		# for word in words:
		# 	if current_length + len(word) + 1 > chunk_size:
		# 		chunks.append(" ".join(current_chunk))
		# 		current_chunk = []
		# 		current_length = 0
		# 	current_chunk.append(word)
		# 	current_length += len(word) + 1
		#
		# if current_chunk:
		# 	chunks.append(" ".join(current_chunk))

		return chunks

	def search_relevant_chunks(self, query, k=5):
		results = self.chroma_client.query(query_texts=[query], n_results=k)
		print(json.dumps(results, indent=4, default=str))
		return results['documents'][0]

	def generate_response(self, query, relevant_chunks):
		context = "\n\n".join(relevant_chunks)
		prompt = f"Context:<br />{context}<br /><br />Question: {query}<br /><br />Answer:"
		return self.generate_text(prompt)
