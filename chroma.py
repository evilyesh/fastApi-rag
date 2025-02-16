import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import InvalidCollectionException
from chromadb.config import Settings  # Добавляем импорт Settings
from typing import List, Dict, Optional
import os

class ChromaDBClient:
    def __init__(self, collection_name: str = "documents", persist_directory: Optional[str] = None):
        # Настройки с разрешением сброса
        self.settings = Settings(allow_reset=True)  # Разрешаем reset

        # Инициализация клиента с настройками
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=self.settings  # Передаем настройки
            )
        else:
            self.client = chromadb.Client(
                settings=self.settings  # Передаем настройки
            )

        self.collection = self._get_or_create_collection(collection_name)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

    def _get_or_create_collection(self, name: str):
        try:
            return self.client.get_collection(name)
        except (ValueError, InvalidCollectionException):
            return self.client.create_collection(name)

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        """
        Добавляет документы в коллекцию.

        :param documents: Список текстовых документов.
        :param metadatas: Список метаданных для каждого документа.
        :param ids: Список идентификаторов для каждого документа.
        """
        if metadatas is None:
            metadatas = [{}] * len(documents)
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_texts: List[str], n_results: int = 5, where: Optional[Dict] = None) -> Dict:
        """
        Выполняет поиск по коллекции.

        :param query_texts: Список текстовых запросов.
        :param n_results: Количество возвращаемых результатов.
        :param where: Фильтр по метаданным.
        :return: Результаты поиска.
        """
        return self.collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where
        )

    def delete(self, ids: Optional[List[str]] = None, where: Optional[Dict] = None):
        """
        Удаляет документы из коллекции.

        :param ids: Список идентификаторов для удаления.
        :param where: Фильтр по метаданным для удаления.
        """
        self.collection.delete(
            ids=ids,
            where=where
        )

    def update(self, ids: List[str], documents: Optional[List[str]] = None, metadatas: Optional[List[Dict]] = None):
        """
        Обновляет документы в коллекции.

        :param ids: Список идентификаторов для обновления.
        :param documents: Новые текстовые документы.
        :param metadatas: Новые метаданные.
        """
        self.collection.update(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def get_collection_info(self) -> Dict:
        """
        Возвращает информацию о коллекции.

        :return: Словарь с информацией о коллекции.
        """
        return {
            "name": self.collection.name,
            "count": self.collection.count()
        }

    def reset(self):
        """
        Удаляет все данные из коллекции.
        """
        self.client.reset()

# Пример использования
if __name__ == "__main__":
    # Инициализация клиента
    chroma_client = ChromaDBClient()

    # Добавление документов
    documents = ["Cats are great pets.", "Dogs are loyal companions.", "Birds can sing beautifully."]
    metadatas = [{"category": "animals"}, {"category": "animals"}, {"category": "animals"}]
    ids = ["doc1", "doc2", "doc3"]
    chroma_client.add_documents(documents, metadatas, ids)

    # Поиск документов
    query_results = chroma_client.query(query_texts=["What are good pets?"], n_results=2)
    print("Результаты поиска:", query_results)

    # Получение информации о коллекции
    print("Информация о коллекции:", chroma_client.get_collection_info())

    # Удаление документа
    chroma_client.delete(ids=["doc1"])
    print("После удаления doc1:", chroma_client.get_collection_info())

    # Обновление документа
    chroma_client.update(ids=["doc2"], documents=["Dogs are the best friends."])
    print("После обновления doc2:", chroma_client.query(query_texts=["Dogs"], n_results=1))

    # # Сброс коллекции
    # chroma_client.reset()
    # try:
    #     print("После сброса:", chroma_client.get_collection_info())
    # except InvalidCollectionException as e:
    #     print("После сброса:")
    #     print('Коллекция не существует!')