# 📄 TestGen – AI-Powered Question Paper Generator

**TestGen** is a smart, AI-driven question paper generation tool designed to create customized exam papers based on previous year questions, topic weightage, user-defined patterns, and difficulty levels. It enables educators and learners to generate personalized, balanced, and efficient assessments using the power of LLMs, FAISS, and RAG (Retrieval-Augmented Generation).



https://github.com/user-attachments/assets/ef09a45a-bafe-4ed9-bb94-323b86bf661a


## 🚀 Features

- 🔍 **Smart Retrieval**: Uses FAISS to fetch the most relevant questions from a large database based on topic, subtopic, and difficulty.
- 🧠 **LLM Integration**: Generates new questions using Gemini or OpenAI based on topic input and weightage distribution.
- ⚖️ **Weightage Handling**: Allows custom weightage input (e.g., 40% PYQ, 60% new) and aligns questions accordingly.
- 🧾 **Pattern Configuration**: Users can define question types (e.g., 5 marks, 10 marks), number of questions, and total marks.
- 🧮 **Marks & Topic Mapping**: Dynamically assigns marks, subtopics, and serial numbers to each question.
- 📊 **Personalized Content Recommendation**: Suggests **YouTube videos and learning resources** based on weak or uncovered areas, identified by gaps in the user’s selected question set.
- 📦 **Structured Output**: Generates clean JSON or dictionary-like output, easy to integrate into UI or export to PDF.

---

## 🛠️ Tech Stack

| Component           | Tech Used                           |
|---------------------|--------------------------------------|
| Backend             | Python, Flask                        |
| LLM Integration     | Gemini API / OpenAI GPT-4            |
| Vector Search       | FAISS + Sentence-BERT                |
| Retrieval Logic     | RAG (Retrieval-Augmented Generation) |
| Data Handling       | Pandas, JSON                         |
| YouTube Recommender | YouTube Search API, Custom Matching |
| Web App (optional)  | Streamlit / Flask Frontend           |

---

## 📚 How It Works

1. **User Input**:
   - Select subject, topics, question type (5/10 marks), number of questions, total marks.
   - Set topic-wise weightage and PYQ vs AI-generated split (e.g., 60:40).

2. **Question Pool Building**:
   - Questions are fetched from past year PDF dumps (parsed and indexed).
   - FAISS retrieves topic-relevant PYQs.
   - LLM generates new questions on uncovered areas based on topic distribution.

3. **Paper Assembly**:
   - Questions are selected and mapped to marks.
   - Subtopics are auto-assigned using semantic understanding.
   - A final dictionary (or JSON) is created: `{"Sr.No": x, "Question": ..., "Subtopic": ..., "Marks": ...}`

4. **Content Recommendation**:
   - Topics with insufficient coverage trigger a YouTube content search.
   - API returns video links to strengthen weak areas.


