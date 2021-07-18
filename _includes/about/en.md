# Seunghwan Hong (홍승환)

-   [Resume](https://bit.ly/2tF0JiZ)
-   [harrydrippin@gmail.com](mailto:harrydrippin@gmail.com)
-   [github.com/harrydrippin](https://github.com/harrydrippin)
-   [linkedin.com/in/harrydrippin](https://linkedin.com/in/harrydrippin)
-   +82 010-4550-9287

## Recent Professional Experiences

### Scatter Lab (Pingpong Team)

> **Machine Learning Engineer (Seoul, Korea)** <br>
> Dec. 2019 ~ Present <br> **Keywords:** PyTorch, TensorFlow, Spring Boot, GCP, AWS, SWIG, C++

- Implement and manage overall ML engineering parts including ML pipeline, serving optimization, data engineering, model optimization, internal tools/libraries.
- Build a pipeline for preprocessing and pseudonymizing 600+GB sized text data, and vector indexing using Kubeflow Pipelines.
    - Build a internal library that collects and manages filters for de-identifying data.
    - Build a pipeline for automatic build and deployment to manage Docker images for pipeline.
    - Build a research system on GCP that enables efficient research while maintaining privacy compliance.
- Optimize a pretraining process of large size language model for various models.
    - Optimize BERT pretraining process with distributed training strategies using 16-32 node cluster above multiple cloud components (Internal distributed training library, EFA, FSx, S3), collaborated with AWS MLSL.
    - Implement training code for training billion-size GPT-2 using DeepSpeed, and data preprocessing code using Apache Beam.
    - Conduct investigation for searching bottlenecks for optimizing Cloud TPU performance while pretraining using Cloud TPU Profiler.
- Conduct a research for multiple vector similarity search frameworks for real-time inference.
    - Build an early version of faiss-serving, server for inferencing vector similarity search above Faiss index using C++.
    - Refactor faiss-serving using multi-threaded worker on Python. Achieved 130 ~ 150 RPS with static memory usage above n-thousand concurrent users, which is 5x faster than early version.
- Implement initial version of Pingpong Flow (inference pipeline of 'Luda Lee’, a conversational chatbot).
    - Build a library for loading MeCab on Java environment, enabling morpheme analysis with custom dictionary inside Spring - Boot project. (github.com/scatterlab/mecab-ko-java)
    - Build a cloud-based log pipeline system to efficiently collect and statistically analyze the various types of logs - from the chatbot pipeline and ML model using BigQuery and Cloud Logging.
- Build a Kubernetes cluster for deploying various internal tools, using Istio and Argo CD.
    - Build a model registry server using ML Metadata (TFX) and deploy to the internal cluster.
- Contribute to the establishment and settlement of an team development culture.
    - Build a team development guide for managing Python project, including contents about linter, CI/CD, commit convention, etc.
    - Lead various study sessions about Docker/Kubernetes and Go.

### Common Computer (AI Network)

> **Software Engineer (Seoul, Korea)** <br>
> Sep. 2018 – Oct. 2019 <br> **Keywords:** Docker, Typescript, Express.js, gRPC, Protocol Buffers, Firebase Realtime Database, AWS S3

-   Built blockchain backend for executing a distributed computing job with ERC-20 token for enabling users to access our system more easily.
-   Implemented gRPC, JSON-RPC based blockchain backend for making communication between Chrome Extension and distributed nodes who are participating in the network as a computing node.
-   Implemented isolated education and machine learning environment for each user using Docker.

### Hyprsense, Inc.

> **Software Engineer Internship (Burlingame, CA)** <br>
> Jun. 2018 – Aug. 2018 <br> **Keywords:** Python, Typescript, Node, WebRTC, OpenCV, Redis, DynamoDB, Terraform

-   Built framework for managing and annotating dataset including Valid Landmark, Tongue, and Face Checker, Misaligned Image Collector, Landmark Annotator, etc.
-   Built deep learning model testing platform based on the web application by WebRTC connection between the web browser and inference server.
-   Implemented interface linkage library for deep learning vision model and a web application by shipping lightweight model directly to the web browser. Video latency was reduced from 1~5 seconds to 6~10 milliseconds.
-   Defined overall infrastructure as a code using Terraform and implemented Python script for managing deployments within the CI/CD pipeline.

### Jininsa Company

> **Software Engineer (Seoul, Korea)** <br>
> Jun. 2017 – Dec. 2017 <br> **Keywords:** Flask, Docker, Redis, Shell Script, Python, SQL

-   Built rule-based chatbot decision system to make the fitting conversation with kids for various situations like singing songs, reading books, end-to-end game, etc.
-   Implemented API services for communication with chatbot server and managing user information which injects user information to reply decision process.
-   Containerized all infrastructures using Docker and made a set of scripts for controlling containers.

## Technical Skills

**Languages:** Javascript (Typescript), Python, Go, C++, Java, SQL <br>
**Frameworks / Platforms:** Docker, Flask, Express.js, gRPC, React.js, Terraform <br>
**Others:** Jupyter Notebook, Web Publishing, Web Crawling, Sketch, Zeplin, Agile Development

## Education

#### University of California, Irvine (Irvine, CA, United States)

Visiting Researcher in Informatics, Software Design and Collaboration Laboratory <br>
Jun. 2019 – Present

#### Kookmin University (Seoul, South Korea)

Graduate in Computer Science, Average in Major: 3.5 / 4.0, 3.94 / 4.5 <br>
Mar. 2016 – Feb. 2020

## Honors and Awards

#### Finalist, HACK/HLTH 2019 by AngelHack, HLTH Conference

Built a FHIR data pipeline for making medical data access permission system. <br>
Apr. 2019

#### Finalist, F8 2019 Hackathon, Facebook

Built a chatbot and web application for tracking various problems on the city. <br>
Apr. 2019

#### Prize for Best Engineering, 23rd Startup Weekend

Created a system for managing hackathon and evaluating Git projects to exact number of scores. <br>
Apr. 2017

#### 1st Place (National Competition), South Korea Government R&D Business

Successfully completed my projects and achieved the overall first grade in this business. <br>
Dec. 2016

#### 4th Place (National Competition), 2015 Korea Olympiad in Informatics

Built new educational programming language which can program with Korean letters. <br>
Oct. 2015

## Contributions and Invited Talks

#### Facebook Developer Circle: Seoul, F8 2019 Meetup

Review Session for F8 2019 Hackathon: Invited Speaker <br>
2019

#### Openhack: National Union of Engineering School Hackathon

Organizer & System Developer <br>
2017

#### Google Developer Group: Korea, Campus Summer Party

Technical Session: Invited speaker (‘Reactive Programming for Beginners’) <br>
2017

## Publications

Case of Social Contribution Project using Open Source Software Development Principle, **Seunghwan Hong**, Gihyeon Yang, Seongkwon Yoon, Dujin Jeong, Jaeyoung Park, Soochurl Shin, Minsuk Lee, Korea Computer Congress 2018 (KCC 2018), Jeju, South Korea, 2018.

Studying OPEG Score Development for Learning Open Source Software Development Practice, **Seunghwan Hong**, Dong-Gyu Kim, Geon Son, Domin Kim, Soochurl Shin, Dujin Jeong, Rina Choi, Minsuk Lee, Korea Computer Congress 2017 (KCC 2017), Busan, South Korea, 2017.

## Volunteering Experience

### Colored by Software

> Lead of Technical Staff <br>
> May. 2016 – Present

-   Non-profit software education voluntary service that educates software to people, mainly teenagers, who are interested in software in the library all over the country.
-   Volunteered as a system developer and project manager with leading 5+ more staffs for building library searching system, lecture searching system, landing page, etc.
