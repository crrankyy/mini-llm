import tiktoken

def main():

    with open("data/basic-data.txt", "r") as f:
        raw_text = f.read()
    
    tokenizer = tiktoken.get_encoding("cl100k_base")

    encoded_text = tokenizer.encode(raw_text)

    enc_sample = encoded_text[50:]

    context_size = 9

    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]

    print("-" * 100)
    print(f"Input tokens  (x): {x}")
    print(f"Target tokens (y):      {y}")
    print("-" * 100)

    print(f"Context-target pairs:")

    for i in range(1, context_size+1):
        context = enc_sample[:i]
        target = enc_sample[i]
        print(context, "---->", target)

    print("")

    for i in range(1, context_size+1):
        context = enc_sample[:i]
        target = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([target]))

    print("-" * 100)

if __name__ == "__main__":
    main()