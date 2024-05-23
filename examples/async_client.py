import openllm, asyncio
client = openllm.AsyncHTTPClient('http://0.0.0.0:3000')
async def main(): assert (await client.health()); print(await client.generate('Explain superconductor to a 5 year old.'))
if __name__ == "__main__": asyncio.run(main())
