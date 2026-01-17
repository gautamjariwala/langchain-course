from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


load_dotenv()


def main():
    information = """ Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2021; as of December 2025, Forbes estimates his net worth to be around US$717 billion.

Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania in Philadelphia, United States, before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.

    """
    summary_template = """
    give the information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """
    prompt_template = PromptTemplate(input_variables = ["information"], template = summary_template)


    # llm = ChatOpenAI(model="gpt-5", temperature=0)

    final_prompt = prompt_template.format(information=information)
    # print("===== FINAL PROMPT SENT TO AI =====")
    # print(final_prompt)
    # print("==================================")

    llm = ChatOpenAI(model="gpt-5", temperature=0)
    
    chain = prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)
    # chain = summary_prompt_template | llm
    # print(chain)
    # response = chain.invoke(input={"information": information})
    # print(response.content)


if __name__ == "__main__":
    main()
