import openai
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=80,
                    help="display a square of a given number")


args = parser.parse_args()
print(f'using port: {args.port}')

openai_api_key = "cmsc-35360"
#openai_api_base = f"http://103.101.203.226:{args.port}/v1"
openai_api_base = f"http://66.55.67.65:{args.port}/v1"
print(openai_api_key)
print(openai_api_base)
print('llama31-405b-fp8')
print("")


client = openai.OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model='llama31-405b-fp8',
    messages=[
        {"role": "user", "content": "Please generate four hypothesis in the origins of life that could be explored with a self-driving laboratory.  For each example please list the key equipment and instruments that would be needed and the experimental protocols that would need to be automated to test the hypotheses."},
    ],
    temperature=0.0,
    max_tokens=2056,
)
#print("Chat response:", chat_response)
print(chat_response.choices[0].message.content)
