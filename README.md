# VTT Meeting Note Generator

**Generate meeting note summaries from Microsoft Teams, Zoom or other .vtt formatted transcripts, using a nominated OpenAI GPT Large Language model running within your own Microsoft Azure tenancy.**

Record Sure Limited (['Recordsure'](https://recordsure.com)) licences this file to you under the [MIT license](LICENSE.txt)

## Overview
The [generateMeetingNote.py](generateMeetingNote.py) file is a Python script that will generate a [summary meeting note](ExampleTranscript/ExampleTranscript1_meeting_note.docx) in Microsoft Word format, given a .vtt ([WebVTT](https://en.wikipedia.org/wiki/WebVTT)) file as its sole input parameter.

Using your own Azure tenancy and therefore your own OpenAI GPT 3.5 'deployment' (for security), once you've used up your free tier credits, then Microsoft will charge you approximately 1.6p (GBP) / 2� (USD) to generate a Meeting Note summary for a 1-hour transcript, based on this code. 

To use it you will need to configure environment variables, or populate the .env file, with details of an OpenAI GPT instance ('deployment') that you create in your Azure Directory, along with your access key.

It will generate its output in a .docx file within the same directory.

You can configure the entries in the [Prompts.xlsx](Prompts.xlsx) file to generate meeting topics relevant to your use case.

## Disclaimers
Whilst fully usable as-is, this project is not provided as hardened (i.e. production-grade) code, and instead is deliberately simplified in order to serve as a simple reference as to how to generate meeting notes in the style that you require.

It is intended for creators and users of such GPT summarisation technology to explore both how easy such customised meeting note generation is (less than 200 lines of code, spaced and commented!), as well as to evaluate the potential accuracy of the results of using GPT technology.

However you use this code, please ensure you advise any potential end-user of your product that the output has been generated by AI, and as such, it must be verified by a human before being used as the basis for taking any action regards a customer or other individual.

The Microsoft charge approximation above is calculated using the Microsoft Azure UK-South region during June 2024, but will vary based on your geography/exchange rates and from time-to-time general price fluctuation from OpenAI/Microsoft.

# Getting Started
You will need to have Python installed and a very basic knowledge of Python. 

If you are not a software developer then don't worry - an hour or so of time from a friendly Python developer whould be enough to get you started with generating your own transcripts, with your own configurations; whilst you'll see from things like the ['Prompts.xlsx'](Prompts.xlsx) configuration file, we've made this as Windows friendly as possible.

Otherwise, here's all you need to know.

## Creating a secure OpenAI GPT instance in your own Azure Directory
The reason we've supplied this code configured for an Azure OpenAI GPT instance ('deployment') is for your security.
You probably don't want your meeting transcripts handed over to OpenAI (or any other party), so using your own Azure instance keeps things secure.

If you have an IT department, then there is very good chance they already have an Azure Directory, so you just need to convince them to give you access to an OpenAI GPT instance you can use.

If not, then start by signing up for Microsoft Azure here: [https://azure.microsoft.com](https://azure.microsoft.com/en-us/free/)

Once in, you (or your IT department) will need to follow [Microsoft's instructions](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource) to create a GPT resource.

When selecting a model to use for your deployment, try and select the newest Pay As You Go (Microsoft 'Deployment Type: Standard') version of either:
* OpenAI GPT-3.5-Turbo (cheaper and works well) 
* OpenAI GPT4 (more expensive and, frankly, unnecessary for this use case)

When going through the Azure OpenAI deployment setup instructions, make sure to take a note of:
* The instance endpoint: it will be a URL, like https://something-unique-to-your-company.openai.azure.com/ 
* Your API key (either of the two generated): it will be a 'UUID' - a seemingly random sequence of 32x letters and numbers
* The GPT model 'Deployment name' you have created (not OpenAI's name for the model).

## Running the generator script
This code has been developed on [Python v3.10.10](https://www.python.org/downloads/release/python-31010/), and whilst newer versions should work, the OpenAI libraries required in particular may not work on earlier Python versions.

Assuming you have Python installed, you will need to install the required dependencies, so in a shell or Windows command prompt (preferably Administrator mode), switch to the download directory (or local Git Repo) for this project and run:
```
pip install -r requirements.txt
```

You will then need to ensure you set the Environment Variables for your Microsoft Azure OpenAI GPT instance. The easiest way to do this is to edit the [.env](.env) file, adding the missing information.
```
OPENAI_API_TYPE="azure" # leave this alone if using a secure Azure Directory rather than OpenAI itself
OPENAI_API_BASE="https://something-unique-to-your-company.openai.azure.com/" 
OPENAI_API_KEY="your-azure-openai-key"
OPENAI_API_VERSION="2023-05-15" # depending on when you run this, the Azure OpenAI version date may have changed
OPENAI_API_DEPLOYMENT_NAME="your-GPT-model-deployment-name" 
```

Then to generate a summary document from a VTT transcript file you have downloaded from Zoom, Microsoft Teams or another source, run:
```
python generateMeetingNote.py C:\path-to\my-WebVTT-file.vtt 
```

If you haven't got your own WebVTT format transcript to play with yet, then we've included a [sample one](ExampleTranscript/ExampleTranscript1.vtt) in the ExampleTranscript folder.

## The [Prompts.xlsx](Prompts.xlsx) configuration file
Before changing this 'configuration' file it helps to have a feel for 'Prompt Engineering' ([Google it](https://www.google.com/search?q=gpt+prompt+engineering)) - so if nothing else have a play with ChatGPT or one of its competitors to understand how to phrase queries.

The spreadsheet file has two worksheets:
* **Topic Prompts**: List as many topics as you like in here, with column A being the topic heading, and column B being the instruction prompt to pass the GPT model as to what to include in each topic.
* **Core Prompts**: Four sets of prompts to pass to the GPT model, each of which can have one or more lines of instruction. You must retain the labels used in column A, however, you can vary the number of lines (rows) for each:
  * _key_points_summarization_: These instructions are used to pick out key information from the transcript, being the various summary bullets generated for the topics. In here you can give examples of the types of information you are more interested in.
  * _overall_summarization_: The prompts used to create the overall 'Conversation Summary' - being a summarised aggregate of all of the summary items found (from the key_points_summarization). 
  * _topic_analysis_prelude_: The opening prompts used to categorise the summary bullets found (from the key_points_summarization) into their topics. It helps to reference the types of parties involved here (e.g. 'financial advisor' and 'customer'). You must leave in the reference to the 'enclosed triple backticks'.
  * _topic_analysis_postlude_: The closing prompts used in the topic categorisation.

That's about it, folks. 
...other than to say that if you do want to productionise the code above and have it automagically generate customised meeting notes from, say, your Microsoft Teams meetings, then perhaps either:
* if you are an enterprise or product supplier who wants to provide/sell a Teams 'bot', then perhaps start with [Microsoft's giveaway code here](https://learn.microsoft.com/en-us/samples/officedev/microsoft-teams-samples/officedev-microsoft-teams-samples-meetings-transcription-nodejs/).
* ignore all of the code above, now that you understand how simple this all is, and simply create a Microsoft Power Automate Flow to grab your meeting notes from Teams and run them through a GPT model, [perhaps starting here](https://learn.microsoft.com/en-us/ai-builder/use-a-custom-prompt-in-flow).

# And Finally
**If you are a [Recordsure Capture or ConversationAI](https://recordsure.com) customer then, frankly, ignore all of the above.** 

**Recordsure provides our clients with editable Meeting Note Summaries _at no extra cost_; as well as evidence citations and more powerful tools for advice efficiency, such as AI summary themes based on _real_ predictive AI, trained by data scientists and human annotators over many years on _real_ financial advice conversations.**

If you'd like to know more about the strengths and limitations of using GPT technologies to generate meeting note summaries from conversations, [read our blog here](https://recordsure.com/resources/insights/).

P.S. please, please, please [use AI Responsibly](https://recordsure.com/responsible-ai/)