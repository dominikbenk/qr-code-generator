# Creating Artistic QR Codes at Scale Using LangChain and ControlNet

## Summary
We built a tool that can generate artistic QR codes for a specific website/url with the use of [DeepLake](https://www.activeloop.ai/), [LangChain](https://python.langchain.com/docs/get_started/introduction.html), [Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [ControlNet](https://github.com/Mikubill/sd-webui-controlnet). 

DeepLake is a multi-modal vector database, designed to efficiently store and search large-scale AI data including audio, video or embeddings from text documents, which will also be utilized in this article. It offers unique storage optimization for deep learning applications, featuring data streaming, vector search, data versioning, and seamless integration with other popular frameworks such as LangChain. This comprehensive toolkit is designed to simplify the process of developing workflows of large language model (LLM), and in our case we will focus on its capability to summarize and answer questions from large-scale documents such as web pages. 

Stable diffusion is a recent development in the field of image synthesis, with exciting potential for reducing high computational demand. It is primarily used for text-to-image generation, but is capable of variety of other tasks such as image modification, inpainting, outpainting, upscaling and generating image-to-image conditioned on text input. Meanwhile, ControlNet is an innovative neural network architecture that is a game-changer in managing the control of these diffusion models by integrating extra conditions. These control techniques include edge and line detection, human poses, image segmentation, depth maps, image styles or simple user scribbles. By applying these techniques, it is then possible to condition our output image with QR codes as well. In case you would be interested in more details, we recommend reading the [original ControlNet article](https://arxiv.org/abs/2302.05543).


By combining all of this, we can achieve a scalable generation of QR codes that are very unique and more likely will attract attention. Overall, there are many possibilities you can approach this problem, and in this article, we will present those that we believe have the highest potential to impact advertising in the future. These are the steps that we are going to walk you through:

## Steps
1. Scraping the Content From a Website and Splitting It Into Documents
2. Saving the Documents Along With Their Embeddings to Deeplake
3. Extracting the Most Relevant Documents
4. Creating Prompts to Generate an Image Based on Documents
    - 4.1 Custom summary prompt + LLMChain
    - 4.2 QA retrieval + LLM
5. Summarizing the Created Prompts
6. Generating Simple QR From URL
7. Generating Artistic QR Codes for Activeloop
    - 7.1. Txt2Img
        - 7.1.1 Content prompt
        - 7.1.2 Portrait prompt
        - 7.1.3 Deeplake prompt
    - 7.2. Img2Img with logo
        - 7.2.1 Content prompt
        - 7.2.2 Portrait prompt
        - 7.2.3 Deeplake prompt
8. Generating Artistic QR Codes for E-commerce
    - 8.1. Img2Img with logo - Tommy Hilfiger
    - 8.2. Img2Img with logo - Patagonia
9. Limitations of Our Approach
10. Conclusion
        
    
Before we start, we need to install requirements, import LangChain and set the following API tokens:
- Apify token - web scraping/crawling
- Activeloop token - vector database
- OpenAI token - language model


``` python
# Install dependencies
## pip install openai langchain deeplake apify-client tiktoken pydantic==1.10.8

# Import libraries
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.utilities import ApifyWrapper
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.base import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
import os

# Set API tokens
os.environ['OPENAI_API_KEY'] = '<YOUR_OPENAI_TOKEN>'
os.environ['ACTIVELOOP_TOKEN'] = '<YOUR_ACTIVELOOP_TOKEN>'
os.environ["APIFY_API_TOKEN"] = '<YOUR_APIFY_TOKEN>'
```


### Step 1: Scraping the Content From a Website and Splitting It Into Documents
First of all, we need to collect data that will be used as a content used to generate QR codes. Since the goal is to personalize it to a specific website, we provide a simple pipeline that can crawl data from a given URL. As an example, we use https://www.activeloop.ai/ from which we scraped 20 pages, but you could use any other website as long as it does not violate the Terms of Use. Or, if you wish to use other type of content, 
LangChain provide many other [File loaders](https://js.langchain.com/docs/modules/indexes/document_loaders/examples/file_loaders/) and [Website loaders](https://js.langchain.com/docs/modules/indexes/document_loaders/examples/web_loaders/) and you can personalize QR codes for them too!

``` python
# We use crawler from ApifyWrapper(), which is available in Langchain
# For convenience, we set 20 maximum pages to crawl with a timeout of 300 seconds.
apify = ApifyWrapper()
loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://www.activeloop.ai/"}], "maxCrawlPages": 20},
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
    ),
    timeout_secs=300,
)

# Now the pages are loaded and split into chunks with a maximum size of 1000 tokens
pages = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator = ".")
docs = text_splitter.split_documents(pages)
```



### Step 2: Saving the Documents Along With Their Embeddings to Deeplake
Once the website is scraped and pages are split into documents, it's time to generate the embeddings and save them to the DeepLake. This means that we can come back to our previously scraped data at any time and don't need to recalculate the embeddings again. To do that, you need to set your `ACTIVELOOP_ORGANIZATION_ID`.

``` python
# initialize the embedding model
embeddings = OpenAIEmbeddings()

# initialize the database, can also be used to load the database
db = DeepLake(
    dataset_path="hub://<ACTIVELOOP_ORGANIZATION_ID>/scraped-websites",
    embedding_function=embeddings,
    token=os.environ['ACTIVELOOP_TOKEN'],
    overwrite=False,
)

# save the documents
db.add_documents(docs)
```

### Step 3: Extracting the Most Relevant Documents
Since we want to generate an image in the context of the given website that can have hundreds of pages, it is useful to filter documents that are the most relevant for our query, in order to save money on chained API calls to LLM. For this, we are going to leverage [Deep Lake Vector Database](https://docs.activeloop.ai/tutorials/vector-store/deep-lake-vector-store-in-langchain) similarity search as well as retrieval functionality.

To pre-filter the documents based on a query, we can do the following

``` python
query = 'You are a prompt generator. Based on the content, write a detailed one sentence description that can be used to generate an image'

result = db.similarity_search(query, k=10)
```

For question-answering pipeline, we can then define the retriever
``` python
retriever = db.as_retriever(
    search_kwargs={"k":10}
)
```

### Step 4: Creating Prompts to Generate an Image Based on Documents
The goal is to understand the content and generate prompts in an automated way, so that the process can be scalable. We start by initializing the LLM with a default `gpt-3.5-turbo` model and set the high temperature to introduce more randomness.
``` python
# Initialize LLM
llm = OpenAI(temperature=0.9)
```

One of many advantages of LangChain are also prompt templates, which significantly help with clarity and readability. To make the output description more precise, we should also provide examples as can be seen here.

``` python
prompt_template = """{query}:

Example:

Content: real estate company
Detailed image description of building: Detailed image description: best quality, realistic, photorealistic, ultra detailed, 8K, High quality texture, intricate details, detailed texture, finely detailed, photo of building partial_view, flower pond, tree, no humans, sky, scenery, outdoors, cloud, blue sky, day, palm tree, plant


Content: {text}
Detailed image description:
"""

# set the prompt template
PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["text"], 
    partial_variables={"query": query}
)

```
The `query` is identical to the one that is used during similarity search and
`text` is the content that is provided to the LLM, based on which it is then supposed to provide a detailed image description. Additionally, to have more control over the output, we also create an alternative prompt that is able to generate specific image type.
``` python
image_type='portrait'

prompt_template = """{query}:

Example:

Content: real estate company
Detailed image description of building: Detailed image description: best quality, realistic, photorealistic, ultra detailed, 8K, High quality texture, intricate details, detailed texture, finely detailed, photo of building partial_view, flower pond, tree, no humans, sky, scenery, outdoors, cloud, blue sky, day, palm tree, plant


Content: {text}
Detailed image description of {image_type}: 
"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["text"], 
    partial_variables={"query": query, "image_type": image_type}
)

```

Using this, we then experimented with 2 following approaches, that differ in what kind of `text` is provided.


#### Option 1: Custom summary prompt with LLMChain

The idea is simple, we chain the description prompt on each filtered document and then apply it once again on the summarized descriptions. In other words, `text` will be a variable that is iterated during `LMMChain` operation.

``` python
# Initialize the chain
chain = LLMChain(llm=llm, prompt=PROMPT)

# Filter the most relevant documents
result = db.similarity_search(query, k=10)
# Run the Chain
chain.run(result)
```
#### Option 2: Retrieval Question-Answering with LLM

Here we initialize QA retriever, which will allow us to ask to explain a particular concept on the filtered documents.
``` python
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type='stuff', 
    retriever=retriever
)

answer = qa.run("Explain what is deeplake")
```
The `answer` is then used as `text` in the `PromptTemplate` without the need for any chain.

``` python
llm(prompt=PROMPT.format(text=answer))
```

### Step 5: Summarizing the Created Prompts
We experimented with different prompt setups in the previous section, and yet there is more to explore. In case you would be interested in perfectionizing your LLM prompts even further, we have an [amazing course](https://learn.activeloop.ai/courses/take/langchain/multimedia/46317727-intro-to-prompt-engineering-tips-and-tricks) that will provide you many useful tips and tricks. Mastering prompts for image generation is, however, more art than science. Nevertheless, by providing the LLM with examples we can see that it can do a pretty good job by generating very specific image descriptions. Here are 3 different types of prompts that we were able to generate with our approach:

#### 1. Content prompt
This prompt summarizes all relevant documents scraped from Activeloop into a general but detailed image description: `high-tech, futuristic, AI-driven, advanced, complex, computer-generated, robot, machine learning, data visualization, interactive, cutting-edge technology, automation, precision, efficiency, innovation, digital transformation, smart technology, science fiction-inspired.
`

#### 2. Portrait prompt
Additionally to previous prompt, we also condition on the type of the image, which is in this case a detailed image description of a portrait: `High quality portrait of a developer working on LangChain, surrounded by computer screens and programming tools, with a focus on the keyboard and coding on the screen.
`

#### 3. DeepLake prompt
Here we show a Question-Answering example with a detailed image description of DeepLake: `An aerial view of a serene, glassy lake surrounded by trees and mountains, with giant blocks of data floating on the surface, each block representing a different data type such as images, videos, audio, and tabular data, all stored as tensors, while a team of data scientists in a nearby cabin focus on their work to build advanced deep learning models, powered by GPUs that are seamlessly integrated with Deep Lake.
`


### Step 6: Generating Simple QR From URL
Before we generate the art, it is important to prepare the simple QR code for ControlNet, which can be done for example [here](https://keremerkan.net/qr-code-and-2d-code-generator/). It is important to set the error correction level to 'H', which increases the probability of QR being readable, as 30% of the code can be covered/destroyed by an image. To generate QR code with a logo, we can then use [this website](https://www.qrcode-monkey.com/) for example. It is also important to note, that some of the URLs might be too long to generate a QR that is not too complicated and reliable enough for scanning. For this purpose, we can use url shorteners such as [bit.ly](bit.ly).

### Step 7: Generating Artistic QR Codes for Activeloop
First of all, we need to keep in mind that it is still very fresh and unexplored topic and the more pleasing-looking QRs you want to generate, the higher risk of not being readable by a scanner. This results in an endless cycle of adjusting parameters to find the most general setup. Many approaches can be applied, but their main difference is in ControlNet units. The highest success we had was with [brightness and tile preprocessors](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main), as well as the [qrcode preprocessor](https://huggingface.co/DionTimmer/controlnet_qrcode). Sometimes, adding a depth preprocessor was also helpful. A great guide on how to set up the Stable-diffusion webui with ControlNet extension to generate your first QR codes can be found for example [here](https://www.youtube.com/watch?v=HOY5J9UT_lY). Nevertheless, there is no single setup that would work 100% of the time and a lot of experimenting is needed, especially in terms of finetuning the control's strength/start/end to achieve a desirable output.

For example, in most of the QR codes we used the following setup:
- Negative prompt: ugly, disfigured, low quality, blurry, nsfw
- Steps: 20
- Sampler: DPM++ 2M Karras
- CFG scale: 9
- Size: 768x768
- Model: dreamshaper_631BakedVae
- ControlNet
    - 0: preprocessor: none, model: control_v1p_sd15_qrcode, weight: 1.1, starting/ending: (0, 1), resize mode: Crop and Resize, pixel perfect: False, control mode: Balanced
    - 1: preprocessor: none, model: control_v1p_sd15_brightness, weight: 0.3, starting/ending: (0, 1), resize mode: Crop and Resize, pixel perfect: False, control mode: Balanced
    
In case of Img2Img, we would also need to put an inpaint mask to disable any changes to the logo.


#### Txt2Img - generating QR code from a simple QR and previously created prompt
##### Content prompt
| | |
|:-------------------------:|:-------------------------:|
|<img width="768" src="images/7.1.1.1.png"> | <img width="768" src="images/7.1.1.2.png">|
|<img width="768" src="images/7.1.1.3.png"> | <img width="768" src="images/7.1.1.4.png">|

##### Portrait prompt
| | |
|:-------------------------:|:-------------------------:|
|<img width="768" src="images/7.1.2.1.png"> | <img width="768" src="images/7.1.2.2.png">|
|<img width="768" src="images/7.1.2.3.png"> | <img width="768" src="images/7.1.2.4.png">|

##### Deeplake prompt
| | |
|:-------------------------:|:-------------------------:|
|<img width="768" src="images/7.1.3.1.png"> | <img width="768" src="images/7.1.3.2.png">|
|<img width="768" src="images/7.1.3.3.png"> | <img width="768" src="images/7.1.3.4.png">|
        
#### Img2Img with logo - generating QR code from a QR with logo and previously created prompt
##### Content prompt
| | |
|:-------------------------:|:-------------------------:|
|<img width="768" src="images/7.2.1.1.png"> | <img width="768" src="images/7.2.1.2.png">|
|<img width="768" src="images/7.2.1.3.png"> | <img width="768" src="images/7.2.1.4.png">|

##### Portrait prompt
| | |
|:-------------------------:|:-------------------------:|
|<img width="768" src="images/7.2.2.1.png"> | <img width="768" src="images/7.2.2.2.png">|
|<img width="768" src="images/7.2.2.3.png"> | <img width="768" src="images/7.2.2.4.png">|

##### Deeplake prompt
| | |
|:-------------------------:|:-------------------------:|
|<img width="768" src="images/7.2.3.1.png"> | <img width="768" src="images/7.2.3.2.png">|
|<img width="768" src="images/7.2.3.3.png"> | <img width="768" src="images/7.2.3.4.png">|



### Step 8: Generating Artistic QR Codes for E-commerce

The idea here is a little different compared to the previous examples in context of [Activeloop](activeloop.com).
Now, we focus on product advertising and we want to generate a QR code only for a single URL and its product. The challenge is to generate QR code, while also keeping the product as similar to the original as possible to avoid misleading information. To do this, we experimented with many preprocessors such as the `tile`, `depth`, `reference_only`, `lineart` or `styles`, but we found most of them too unreliable and far from being similar to the original input. At this moment, we believe that the most useful  is the `tile` preprocessor, which can preserve a lot of information. The disadvantage is, however, that it does not allow for many changes during control phase and the QR fit can sometimes be questionable. In practice, this would be done by adding another CotntrolNet unit:
- 2: preprocessor: none, model: control_v11f1e_sd15_tile, weight: 1.0, starting/ending: (0, 1), resize mode: Crop and Resize, pixel perfect: False, control mode: Balanced
Since the `tile` input image control is very strong, theres not much else we can do. Styles are one of the little extra adjustments possible and very useful style cheat sheet can be found [here](https://supagruen.github.io/StableDiffusion-CheatSheet/). For our purposes, however, we did not end up utilizing any of them. 


Similarly as before, we generated prompts automaticaly from the given websites. We randomly selected 2 products and in the first case (Tommy Hilfiger) We added logo to the initial basic QR code while in the second case (Patagonia), we only mask the logo that is already present on the product. To see the comparison, we also provide the original input images (Sources: [Patagonia](https://eu.patagonia.com/cz/en/product/mens-capilene-cool-daily-graphic-shirt/45235.html?dwvar_45235_color=SSMX&cgid=mens-shirts-tech-tops), [Tommy Hilfiger](https://uk.tommy.com/tommy-hilfiger-x-vacation-flag-embroidery-t-shirt-mw0mw33438ybl)).

#### Img2Img with logo - generating Tommy Hilfiger QR code

| | |
|:-------------------------:|:-------------------------:|
| <img width="768" src="images/8.1.1.png"> | <img width="768" src="images/8.1.2.png"> |
| <img width="768" src="images/8.1.3.png"> | <img width="768" src="images/8.1.4.png"> |

#### Img2Img with logo - generating Patagonia QR code
| | |
|:-------------------------:|:-------------------------:|
| <img width="768" src="images/8.2.1.png"> | <img width="768" src="images/8.2.2.png"> |
| <img width="768" src="images/8.2.3.png"> | <img width="768" src="images/8.2.4.png"> |


### Limitations of Our Approach
- Overall, the ControlNet model required extensive manual tuning of parameters. There are many methods to control the QR code generation process, but none are entirely reliable. The problem intensifies when you want to account for the input product image as well. To the best of our knowledge, no other publication has found a way to generate them reliably, and we spent the majority of our time experimenting with various setups.

- Adding an image to the input might offer more control and bring about various use-cases, but it significantly restricts the possibilities of stable diffusion. This usually only results in changes to the image's style without fitting much of the QR structure. Moreover, we saw greater success with text-to-image compared to image-to-image with logo masks. However, the former wasn't as desirable because we believe logos are essential in product QR codes.

- From our examples, it's evident that the generated products don't exactly match the actual products one-to-one. If the goal is to advertise a specific product, even a minor mismatch could be misleading. Nonetheless, we believe that [LORA](https://stable-diffusion-art.com/lora/) models or a different type of preprocessor model could address these issues.

- Automated image prompts can sometimes be confusing, drawing focus to unimportant details within the context. This is particularly problematic if we don't have enough relevant textual information to build upon. This presents an opportunity to further use the DeepLake's vector DB to analyze the image bind embeddings for a better understanding of the content on e-commerce websites.

- In our examples, we also encountered issues with faces, as they sometimes didn't appear human. However, this could be easily addressed with further processing. In instances where we want to preserve the face and adjust it to the QR code, there are tools like the [Roop](https://github.com/s0md3v/sd-webui-roop) that can be used for a detailed face replacement.


### Conclusion: Scalable Prompt Generation Achieved, QR Code Generation Remains Unreliable
DeepLake combined with LangChain can significantly reduce the costs of analyzing the contents of a website to provide image descriptions in a scalable way. Thanks to the vector database, we can save a large number of documents and images along with their embeddings. This allows us to iteratively adjust the image prompts and efficiently filter based on embedding similarities. However, it is very difficult to find the ControlNet sweet spot of QR readability and "cool" design. Taking into account all of the limitations we've discussed, we believe that there needs to be more experimenting with ControlNet, in order to generated product QR codes that are reliable and applicable for real-world businesses.

I hope that you find this useful and already have many ideas on how to further build on this. Thank you for reading and I wish you a great day and see you in the next one.

