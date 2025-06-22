from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
import fitz
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser

load_dotenv(find_dotenv(), override=True)


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
)

data = {
    'syllabus_text': 'What is ArƟﬁcial Intelligence? \nWhat is ArƟﬁcial Intelligence? In today’s rapidly advancing technological landscape, AI has become a household \nterm. From chatbots and virtual assistants to self-driving cars and recommendaƟon algorithms, the impact of AI is \nubiquitous. But what exactly is AI and how does it work? \nAt its core, ArƟﬁcial Intelligence refers to the simulaƟon of human intelligence in machines that are programmed to \nthink, reason, and learn like humans. Rather than being explicitly programmed for speciﬁc tasks, AI(ArƟﬁcial \nIntelligence) systems use algorithms and vast amounts of data to recognize paƩerns, make decisions, and improve \ntheir performance over Ɵme. \nArƟﬁcial Intelligence encompasses a wide range of technologies, including machine learning, natural language \nprocessing, computer vision, and roboƟcs. These technologies enable AI systems to perform complex tasks, such as \nspeech recogniƟon and face detecƟon, with remarkable accuracy. \nIn this arƟcle, we will delve into the intricacies of ArƟﬁcial Intelligence, exploring its various applicaƟons across \nindustries, its potenƟal beneﬁts and challenges, and the ethical consideraƟons surrounding its use. So, join us as we \nunravel the mysteries of AI and its transformaƟve power in our world today. \nTable of Content \n\uf0b7 \nHistory and EvoluƟon of ArƟﬁcial Intelligence (AI) \n\uf0b7 \nCore Concepts in AI \n\uf0b7 \nHow Does AI Work? \n\uf0b7 \nTypes of AI (ArƟﬁcial Intelligence) \n\uf0b7 \nApplicaƟon of ArƟﬁcial Intelligence  \n\uf0b7 \nNeed for ArƟﬁcial Intelligence – Why is AI Important? \n\uf0b7 \nChallenges in ArƟﬁcial Intelligence \n\uf0b7 \nEthical ConsideraƟons in ArƟﬁcial Intelligence \n\uf0b7 \nThe Future of ArƟﬁcial Intelligence \nHistory and EvoluƟon of ArƟﬁcial Intelligence (AI) \nThe concept of ArƟﬁcial Intelligence (AI) has been around for centuries, with the earliest recorded ideas daƟng back \nto ancient Greek mythology. However, the modern ﬁeld of AI emerged in the 1950s, when computer scienƟsts and \nresearchers began exploring the possibility of creaƟng machines that could think, learn, and solve problems like \nhumans. \nOne of the pioneering ﬁgures in the ﬁeld of AI was Alan Turing, a BriƟsh mathemaƟcian and computer scienƟst, who \nin 1950 proposed the Turing test, a method for determining whether a machine can exhibit intelligent behavior \nindisƟnguishable from a human. This sparked a wave of research and development in AI, with scienƟsts and \nresearchers working to create machines that could perform tasks such as playing chess, solving mathemaƟcal \nproblems, and understanding natural language. \nOver the decades, the ﬁeld of AI has evolved signiﬁcantly, with the development of various techniques and \ntechnologies, such as machine learning, deep learning, and natural language processing. The 1980s and 1990s saw a \nsurge in the popularity of expert systems, which were designed to mimic the decision-making process of human \nexperts. In the 2000s, the rise of big data and powerful compuƟng resources paved the way for the development of \nmore advanced AI systems, leading to breakthroughs in areas like computer vision, speech recogniƟon, and \nautonomous vehicles. \nCore Concepts in AI \nArƟﬁcial Intelligence (AI) operates on a core set of concepts and technologies that enable machines to perform tasks \nthat typically require human intelligence. Here are some foundaƟonal concepts: \n1. Machine Learning (ML): This is the backbone of AI, where algorithms learn from data without being explicitly \nprogrammed. It involves training an algorithm on a data set, allowing it to improve over Ɵme and make \npredicƟons or decisions based on new data. \n2. Neural Networks: Inspired by the human brain, these are networks of algorithms that mimic the way \nneurons interact, allowing computers to recognize paƩerns and solve common problems in the ﬁelds of AI, \nmachine learning, and deep learning. \n3. Deep Learning: A subset of ML, deep learning uses complex neural networks with many layers (hence \n“deep”) to analyze various factors of data. This is instrumental in tasks like image and speech recogniƟon. \n4. Natural Language Processing (NLP): NLP involves programming computers to process and analyze large \namounts of natural language data, enabling interacƟons between computers and humans using natural \nlanguage. \n5. RoboƟcs: While oŌen associated with AI, roboƟcs merges AI concepts with physical components to create \nmachines capable of performing a variety of tasks, from assembly lines to complex surgeries. \n6. CogniƟve CompuƟng: This AI approach mimics human brain processes to solve complex problems, oŌen \nusing paƩern recogniƟon, NLP, and data mining. \n7. Expert Systems: These are AI systems that emulate the decision-making ability of a human expert, applying \nreasoning capabiliƟes to reach conclusions. \nEach of these concepts helps to build systems that can automate, enhance, and someƟmes outperform human \ncapabiliƟes in speciﬁc tasks. \nHow Does AI Work? \nArƟﬁcial intelligence (AI) enables machines to learn from data and recognize paƩerns in it, to perform tasks more \neﬃciently and eﬀecƟvely. AI works in ﬁve steps: \n\uf0b7 \nInput: Data is collected from various sources. This data is then sorted into categories. \n\uf0b7 \nProcessing: The AI sorts and deciphers the data using paƩerns it has been programmed to learn unƟl it \nrecognizes similar paƩerns in the data. \n\uf0b7 \nOutcomes: The AI can then use those paƩerns to predict outcomes. \n\uf0b7 \nAdjustments: If the data sets are considered a “fail,” AI learns from that mistake, and the process is repeated \nagain under diﬀerent condiƟons. \n\uf0b7 \nAssessments: In this way, AI is constantly learning and improving. \nTypes of AI (ArƟﬁcial Intelligence) \n1. Narrow AI (ANI) : Narrow AI, also known as ArƟﬁcial Narrow Intelligence (ANI), refers to AI systems designed \nto handle a speciﬁc task or a limited range of tasks. These systems operate under constrained and predeﬁned \ncondiƟons, excelling in their speciﬁc domains but lacking the ability to perform beyond their programmed \ncapabiliƟes. \n2. General AI (AGI) : General AI, or ArƟﬁcial General Intelligence, refers to AI systems that possess the ability to \nunderstand, learn, and apply intelligence across a broad range of tasks, mirroring human cogniƟve abiliƟes. \nAGI can theoreƟcally apply learned knowledge to solve novel problems and perform tasks involving general \nreasoning without prior training speciﬁcally for those tasks. \n3. Superintelligent AI (ASI) : Superintelligent AI, or ArƟﬁcial Superintelligence, represents an AI that not only \nmimics but signiﬁcantly surpasses human intelligence across all ﬁelds — science, general wisdom, social \nskills, and more. ASI would be capable of extraordinary problem-solving and creaƟve abiliƟes, far beyond \nwhat current human minds can achieve.. \nApplicaƟon of ArƟﬁcial Intelligence  \nArƟﬁcial Intelligence has many pracƟcal applicaƟons across various industries and domains, including: \n \n1. Healthcare – AI is used for medical diagnosis by analyzing medical images like X-rays and MRIs to idenƟfy \ndiseases. For instance, AI systems are being developed to detect skin cancer from images with high accuracy. \n2. Finance – AI helps in credit scoring by analyzing a borrower’s ﬁnancial history and other data to predict their \ncreditworthiness. This helps banks decide whether to approve a loan and at what interest rate. \n3. Retail – AI is used for product recommendaƟons by analyzing your past purchases and browsing behavior to \nsuggest products you might be interested in. For example, Amazon uses AI to recommend products to \ncustomers on their website. \n4. Manufacturing – AI helps in quality control by inspecƟng products for defects. AI systems can be trained to \nidenƟfy even very small defects that human inspectors might miss. \n5. TransportaƟon – AI is used for autonomous vehicles by developing self-driving cars that can navigate roads \nwithout human input. Companies like Waymo and Tesla are developing self-driving car technology. \n6. Customer service – AI-powered chatbots are used to answer customer quesƟons and provide support. For \ninstance, many banks use chatbots to answer customer quesƟons about their accounts and transacƟons. \n7. Security – AI is used for facial recogniƟon by idenƟfying people from images or videos. This technology is \nused for security purposes, such as idenƟfying criminals or unauthorized individuals. \n8. MarkeƟng – AI is used for targeted adverƟsing by showing ads to people who are most likely to be interested \nin the product or service being adverƟsed. For example, social media companies use AI to target ads to users \nbased on their interests and demographics. \n9. EducaƟon – AI is used for personalized learning by tailoring educaƟonal content to the individual needs of \neach student. For example, AI-powered tutoring systems can provide students with personalized instrucƟon \nand feedback. \nNeed for ArƟﬁcial Intelligence – Why is AI Important? \nThe widespread adopƟon of ArƟﬁcial Intelligence (AI) has brought about numerous beneﬁts and advantages across \nvarious industries and aspects of our lives. Here are some of the key beneﬁts of AI: \n1. Improved Eﬃciency and ProducƟvity: AI-powered systems can perform tasks with greater speed, accuracy, \nand consistency than humans, leading to improved eﬃciency and producƟvity in various industries. This can \nresult in cost savings, reduced errors, and increased output. \n2. Enhanced Decision-Making: AI algorithms can analyze large amounts of data, idenƟfy paƩerns, and make \ninformed decisions faster than humans. This can be parƟcularly useful in ﬁelds such as ﬁnance, healthcare, \nand logisƟcs, where Ɵmely and accurate decision-making is criƟcal. \n3. PersonalizaƟon and CustomizaƟon: AI-powered systems can learn from user behavior and preferences to \nprovide personalized recommendaƟons, content, and experiences. This can lead to increased customer \nsaƟsfacƟon and loyalty, as well as improved targeƟng and markeƟng strategies. \n4. AutomaƟon of RepeƟƟve Tasks: AI can be used to automate repeƟƟve, Ɵme-consuming tasks, freeing up \nhuman resources to focus on more strategic and creaƟve work. This can lead to cost savings, reduced errors, \nand improved work-life balance for employees. \n5. Improved Safety and Risk MiƟgaƟon: AI-powered systems can be used to enhance safety in various \napplicaƟons, such as autonomous vehicles, industrial automaƟon, and medical diagnosƟcs. AI algorithms can \nalso be used to detect and miƟgate risks, such as fraud, cybersecurity threats, and environmental hazards. \n6. Advancements in ScienƟﬁc Research: AI can assist in scienƟﬁc research by analyzing large datasets, \ngeneraƟng hypotheses, and acceleraƟng the discovery of new insights and breakthroughs. This can lead to \nadvancements in ﬁelds such as medicine, climate science, and materials science. \n7. Enhanced Human CapabiliƟes: AI can be used to augment and enhance human capabiliƟes, such as \nimproving memory, cogniƟve abiliƟes, and decision-making. This can lead to improved producƟvity, \ncreaƟvity, and problem-solving skills. \nWhile the beneﬁts of AI are numerous, it is important to consider the potenƟal challenges and limitaƟons of the \ntechnology, as well as the ethical implicaƟons of its use. \nChallenges in ArƟﬁcial Intelligence \nWhile ArƟﬁcial Intelligence (AI) has brought about numerous beneﬁts and advancements, it also faces several \nchallenges and limitaƟons that must be addressed. Here are some of the key challenges and limitaƟons of AI: \n1. Data Availability and Quality: AI systems rely on vast amounts of high-quality data to learn and make \naccurate predicƟons. However, obtaining and curaƟng such data can be a signiﬁcant challenge, parƟcularly in \ndomains where data is scarce or diﬃcult to collect. \n2. Bias and Fairness: AI algorithms can perpetuate and amplify biases present in the data used to train them, \nleading to decisions and outputs that are unfair or discriminatory. Addressing algorithmic bias is a crucial \nchallenge in the development and deployment of AI systems. \n3. Interpretability and Explainability: Many modern AI systems, such as deep learning models, are complex and \nopaque, making it diﬃcult to understand how they arrive at their decisions. This lack of interpretability can \nbe a signiﬁcant barrier to trust and adopƟon, parƟcularly in sensiƟve domains like healthcare and ﬁnance. \n4. Safety and Robustness: AI systems can be vulnerable to adversarial aƩacks, where small, impercepƟble \nchanges to the input can cause the system to make erroneous or even dangerous decisions. Ensuring the \nsafety and robustness of AI systems is a criƟcal challenge. \n5. Privacy and Security: The collecƟon and use of personal data by AI systems raises signiﬁcant privacy \nconcerns, especially as the technology becomes more pervasive. Balancing the beneﬁts of AI with the need \nto protect individual privacy is an ongoing challenge. \n6. Scalability and ComputaƟonal LimitaƟons: Some AI algorithms and models can be computaƟonally \nintensive, requiring signiﬁcant compuƟng power and resources. Scaling these systems to larger-scale \napplicaƟons can be a challenge, parƟcularly in resource-constrained environments. \n7. Ethical ConsideraƟons: The development and deployment of AI systems raise complex ethical quesƟons, \nsuch as the impact on employment, the accountability for AI-driven decisions, and the potenƟal for AI to be \nused for malicious purposes. Addressing these ethical concerns is crucial for the responsible and trustworthy \nuse of AI. \nAs the ﬁeld of AI conƟnues to evolve, researchers and pracƟƟoners must work to address these challenges and \nlimitaƟons, ensuring that the technology is developed and deployed in a responsible and ethical manner. \nEthical ConsideraƟons in ArƟﬁcial Intelligence \nAs ArƟﬁcial Intelligence (AI) becomes increasingly ubiquitous in our lives, it is crucial to consider the ethical \nimplicaƟons of its development and deployment. Here are some of the key ethical consideraƟons surrounding AI: \n1. Transparency and Accountability: AI systems can be complex and opaque, making it diﬃcult to understand \nhow they arrive at their decisions. This lack of transparency can be problemaƟc, as it can lead to biased or \nunfair outcomes that are diﬃcult to explain or jusƟfy. Ensuring transparency and accountability in AI systems \nis essenƟal for building trust and miƟgaƟng potenƟal harm. \n2. Bias and Fairness: AI algorithms can perpetuate and amplify biases present in the data used to train them, \nleading to decisions and outputs that discriminate against certain individuals or groups. Addressing \nalgorithmic bias and ensuring the fairness of AI systems is a criƟcal ethical challenge. \n3. Privacy and Data Rights: The collecƟon and use of personal data by AI systems raises signiﬁcant privacy \nconcerns, parƟcularly as the technology becomes more pervasive. Balancing the beneﬁts of AI with the \nprotecƟon of individual privacy rights is an ongoing ethical dilemma. \n4. Impact on Employment: The increasing automaƟon of tasks and jobs by AI systems raises concerns about the \npotenƟal displacement of human workers. Addressing the ethical implicaƟons of AI-driven job loss and \nensuring the fair distribuƟon of the beneﬁts of AI is a crucial consideraƟon. \n5. Autonomous Decision-Making: AI systems are being used to make decisions that can have signiﬁcant \nimpacts on people’s lives, such as in healthcare, ﬁnance, and criminal jusƟce. The ethical implicaƟons of \ndelegaƟng decision-making authority to AI systems, parƟcularly in high-stakes scenarios, must be carefully \nexamined. \n6. Misuse and Malicious Use: AI can be used for malicious purposes, such as creaƟng deepfakes, automaƟng \ncyberaƩacks, or enhancing surveillance and control. MiƟgaƟng the potenƟal for the misuse of AI is an \nessenƟal ethical concern. \n7. Societal Impact and Inequality: The widespread adopƟon of AI has the potenƟal to exacerbate exisƟng social \nand economic inequaliƟes, as the beneﬁts of the technology may not be evenly distributed. Addressing the \nethical implicaƟons of the unequal impact of AI is crucial for ensuring the technology beneﬁts society as a \nwhole. \nTo address these ethical consideraƟons, policymakers, researchers, and pracƟƟoners must work together to develop \nethical frameworks, guidelines, and regulaƟons that ensure the responsible development and deployment of AI. This \nincludes promoƟng transparency, accountability, fairness, and the protecƟon of fundamental human rights. \nThe Future of ArƟﬁcial Intelligence \nThe future of ArƟﬁcial Intelligence (AI) is both exciƟng and complex, with the potenƟal to transform virtually every \naspect of our lives. As the technology conƟnues to evolve, we can expect to see a range of advancements and \ndevelopments that will shape the years to come. \n1. Advancements in Machine Learning and Deep Learning: The rapid progress in machine learning and deep \nlearning techniques will enable the creaƟon of even more sophisƟcated and capable AI systems. This includes \nthe development of more accurate and eﬃcient algorithms for tasks such as natural language processing, \ncomputer vision, and predicƟve analyƟcs. \n2. Expansion of Autonomous Systems: The use of AI in autonomous systems, such as self-driving cars, drones, \nand roboƟc assistants, is expected to grow signiﬁcantly. As the technology becomes more reliable and safer, \nwe can expect to see these systems become more prevalent in our daily lives, transforming the way we \ntravel, work, and interact with our surroundings. \n3. Emergence of General AI: While current AI systems are primarily focused on narrow, specialized tasks, the \nlong-term goal of researchers is to develop general AI – systems that can match or exceed human intelligence \nand adaptability across a wide range of cogniƟve tasks. The realizaƟon of general AI would represent a \nsigniﬁcant milestone in the ﬁeld and could lead to transformaƟve breakthroughs in various domains. \n4. IntegraƟon with Internet of Things (IoT) and Edge CompuƟng: As the number of connected devices and \nsensors conƟnues to grow, the integraƟon of AI with IoT and edge compuƟng will become increasingly \nimportant. This will enable the deployment of AI-powered applicaƟons and services at the edge, closer to the \nsource of data, leading to faster response Ɵmes, improved privacy, and reduced reliance on cloud \ninfrastructure. \n5. Advancements in Natural Language Processing and ConversaƟonal AI: The conƟnued progress in natural \nlanguage processing and conversaƟonal AI will enable the development of more natural and intuiƟve \ninterfaces between humans and machines. This could lead to the creaƟon of virtual assistants, chatbots, and \nother AI-powered interfaces that can understand and respond to human language in more meaningful and \ncontextual ways. \n6. Ethical and Regulatory ConsideraƟons: As AI becomes more pervasive, the need for robust ethical \nframeworks and regulatory oversight will become increasingly important. Policymakers, researchers, and \nindustry leaders will need to work together to address issues such as algorithmic bias, privacy, transparency, \nand the societal impact of AI. \n7. Interdisciplinary CollaboraƟon: The future of AI will require close collaboraƟon between various disciplines, \nincluding computer science, cogniƟve science, neuroscience, and ethics. This cross-pollinaƟon of ideas and \nexperƟse will be crucial for addressing the complex challenges and opportuniƟes presented by the \ntechnology. \nAs the future of AI unfolds, we can expect to see a conƟnued acceleraƟon of technological advancements, as well as \nthe emergence of new ethical and societal consideraƟons. By embracing the potenƟal of AI while addressing its \nchallenges, we can unlock new fronƟers of innovaƟon and progress that will shape the world of tomorrow. \nConclusion \nArƟﬁcial intelligence (AI) is revoluƟonizing our world. AI automates tasks, improves decision-making through data \nanalysis, and fuels scienƟﬁc advancements. From healthcare and ﬁnance to transportaƟon and educaƟon, AI has the \npotenƟal to signiﬁcantly enhance our quality of life. \nHowever, responsible development is criƟcal. AI can lead to job displacement and raise ethical concerns about bias in \nalgorithms and privacy issues. Open communicaƟon and collaboraƟon among researchers, developers, policymakers, \nand the public are essenƟal. By harnessing AI’s power for good and focusing on human well-being, we can ensure AI \nbeneﬁts all of humanity. \n \n',
    'syllabus_key_points': ['Artificial Intelligence (AI)', 'Machine Learning (ML)', 'Deep Learning', 'Natural Language Processing (NLP)', 'Applications of AI'],
    'weightage_tags': ['Artificial Intelligence', 'Machine Learning', 'Natural Language Processing'],
    'sections': [{
        'type': 'Answer the following',
        'count': 4,
        'marks': 4
    }, {
        'type': 'MCQ',
        'count': 4,
        'marks': 2
    }]
}
doc = data['syllabus_text']

answersheet = {
  "Section 1": {
    "questions": [
      {
        "question_no": 1,
        "question": "Define Artificial Intelligence (AI) in your own words, highlighting its core functionality.",
        "marks": 2,
        "student_answer": "At its core, AI refers to computer systems capable of performing tasks that typically require human intelligence, such as reasoning, learning, perception and language understanding. These systems analyse vast datasets, recognize patterns and make decisions with unprecedented speed and accuracy."
      },
      {
        "question_no": 2,
        "question": "What is the primary role of Machine Learning (ML) in the context of AI?",
        "marks": 2,
        "student_answer": "Machine learning (ML) is a branch of artificial intelligence (AI) focused on enabling computers and machines to imitate the way that humans learn, to perform tasks autonomously, and to improve their performance and accuracy through experience and exposure to more data."
      },
      {
        "question_no": 3,
        "question": "Explain the relationship between Deep Learning and Machine Learning.",
        "marks": 2,
        "student_answer": "Machine Learning (ML) and Deep Learning (DL) are two subsets of Artificial Intelligence (AI) that are often used interchangeably but have distinct differences in their methodologies, complexity, and applications.\n\nMachine Learning\n\nMachine Learning is a subfield of AI that focuses on developing algorithms and statistical models that enable computers to learn from and make predictions or decisions based on data. It involves training algorithms on large datasets to identify patterns and relationships, which are then used to make predictions or decisions about new data\n1\n.\n\nTypes of Machine Learning\n\nSupervised Learning: Uses labeled data to train the model.\n\nUnsupervised Learning: Finds patterns or groups in data without labels."
      },
      {
        "question_no": 4,
        "question": "Name two real-world applications of Natural Language Processing (NLP).",
        "marks": 2,
        "student_answer": "Sentiment analysis, chatbots, text extraction, text summarization, and speech recognition are some real-life applications of NLP."
      },
      {
        "question_no": 5,
        "question": "Briefly describe the five steps involved in how AI works.",
        "marks": 2,
        "student_answer": "How AI Works: Step-by-Step Process\nStep 1 : Data Collection and Preprocessing\nStep 2: Data Splitting\nStep 3: Model Selection\nStep 4: Model Training\nStep 5: Model Evaluation and Tuning\nStep 6: Model Deployment\nStep 7: Model Monitoring and Maintenance"
      }
    ]
  }
}


def text_splitting(text):
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    return splitter.split_text(text)

def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    documents = [Document(page_content=chunk) for chunk in chunks]
    return FAISS.from_documents(documents, embeddings)

def create_retriever(vector_store, query):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever.invoke(query)

def build_user_query(data: dict) -> str:
    query_parts = []

    # Add topic-based summaries (if available)
    if "syllabus_key_points" in data and data["syllabus_key_points"]:
        query_parts.append(f"Syllabus topics: {', '.join(data['syllabus_key_points'])}")
    
    if "notes_key_points" in data and data["notes_key_points"]:
        query_parts.append(f"Notes highlights: {', '.join(data['notes_key_points'])}")

    if "weightage_tags" in data and data["weightage_tags"]:
        query_parts.append(f"Focus areas: {', '.join(data['weightage_tags'])}")

    # Construct section-specific instructions
    if "sections" in data and data["sections"]:
        section_descriptions = []
        for idx, section in enumerate(data["sections"], start=1):
            section_descriptions.append(
                f"Section {idx}: Generate {section['count']} '{section['type']}' type questions of {section['marks']} marks each"
            )
        query_parts.append("\n".join(section_descriptions))

    # Combine all parts
    user_question = "\n".join(query_parts)

    return user_question

def evaluate_answers(answersheet):
    prompt = PromptTemplate(
        template = """
        You are an expert answer evaluator. You are provided with a structured answer sheet where each question has:
        - the original question
        - the student's answer
        - the number of marks assigned
        - relevant context extracted from notes/syllabus

        Your task is to evaluate each answer based on:
        - Relevance to the context provided
        - Completeness, correctness, and clarity of the response
        - Technical accuracy based on standard knowledge

        ### Evaluation Requirements:

        1. **For each question**, do the following:
        - Evaluate the student's answer.
        - Assign marks out of the given marks (fractional marks are allowed but only in terms of 0.5 and .0).
        
        2. After all evaluations, generate:
        - total_score out of total_possible_marks
        - 3-4 Areas of Improvement (in detailed sentences)
        - 3-4 Suggestions (tips or practices to improve)
        - 4-5 Weak Topics as specific technical terms or domain-specific concepts (e.g., "Machine Learning applications", "Database Transactions", "TCP/IP Layers"). Avoid vague or general skills like "clarity","concise writing", or "understanding the question".
        These weak topics will be used to recommend learning resources, so they must be study topics only, derived either from the question or the provided context.

        
        ### Output Format (Python dictionary/JSON):
        {{
        "score": "X / Y",
        "improvement_areas": [
            "The student struggles to articulate precise advantages of certain technologies."
        ],
        "suggestions": [
            "Revise core concepts using summary notes or flashcards."
        ],
        "weak_topics": ["Machine learning applications", "AI Agents","Python asyncio","PCA component"]
        }}
        
        Use ONLY the context provided to validate correctness, but if the student's answer is independently correct based on general knowledge, still reward them.

        Avoid giving marks for vague or incorrect answers. Be strict but fair.

        Now, evaluate this answersheet:
        {answersheet}
        """,
        input_variables=["answersheet"]
        )
    chain = prompt | llm | JsonOutputParser()
    return chain.invoke({"answersheet":answersheet})

def enrich_answersheet_with_context(answersheet, vector_store, top_k=4):
    """
    For each question in the answersheet, retrieve relevant context from the vector store
    and append it under the 'context' key.
    """
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    for section_data in answersheet.values():
        for q in section_data['questions']:
            # Construct query from question and optionally student answer
            query = f"{q['question']} Student's answer: {q['student_answer']}"
            
            # Retrieve relevant documents
            docs = retriever.invoke(query)
            
            # Join document chunks as a single string
            context_text = "\n".join([doc.page_content for doc in docs])
            
            # Append to question
            q['context'] = context_text

    return answersheet




chunks = text_splitting(doc)
vector_store = create_vector_store(chunks)
#user_query = build_user_query(data)
#context = create_retriever(vector_store, user_query)
#result = generate_questions(context, user_query)
#print(result)
context_answersheet = enrich_answersheet_with_context(answersheet,vector_store)
result = evaluate_answers(context_answersheet)
print(result)




