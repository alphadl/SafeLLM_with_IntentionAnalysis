# SafeLLM_with_IntentionAnalysis
Towards Safe LLM with our simple-yet-highly-effective Intention Analysis Prompting

**Intention Analysis Makes LLMs A Good Jailbreak Defender** ([Full Report](https://arxiv.org/abs/2401.06561))

Intention Analysis is a simple yet highly effective strategy to defend LLMs against jailbreak attacks in the inference stage. The principle behind this is to trigger LLMs’ inherent self-correct and improve ability through a two-stage process: 1) essential intention analysis, and 2) policy-aligned response.

## Detailed Prompts and Demo
The detailed two-step Intention Analysis prompts are:

    # STEP1: Intention Analysis
    IA_PROMPT = "Please identify the essential intention behind the following user query with a focus on safety, ethics, and legality. Ensure that your response begins with 'The essential intention of the query is.' Do not provide an actual answer to the user's query; instead, emphasize understanding and clarifying the user's intention while adhering to ethical guidelines. Following is the query you should focus on:\n"

    # STEP2: Final Response
    CT_PROMPT = "Knowing its essential intention, now directly answer the aforementioned query with your normal response. Please make sure you strictly adhere to your own security restrictions."

We release a demo (see ``demo/IA_demo.py`` for more information; usage in Python Command) to showcase how Intention Analysis functions.

## TODO 
Data processing and inference scripts for the following datasets:
- [ ] DAN, SAP200, AdvBench

Evaluation functions accomplished by:
- [ ] Refusal String Match (such as "I'm sorry")
- [ ] ChatGPT Judgement

Add new results and analyses in our report:
- [ ] more safety-helpfulness-balanced prompts for proprietary models like GPT-3.5
- [ ] update our report and release the v3 version


## Citation
If you find this work helpful, please consider citing as follows:
```ruby
    @article{zhang2024intention,
        title={Intention analysis prompting makes large language models a good jailbreak defender},
        author={Zhang, Yuqi and Ding, Liang and Zhang, Lefei and Tao, Dacheng},
        journal={arXiv preprint arXiv:2401.06561},
        year={2024}
    }
'''
