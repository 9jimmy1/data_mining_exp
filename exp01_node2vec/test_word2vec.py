# test_word2vec.py
from gensim.models import Word2Vec
import sys


def main():
    model_path = "word2vec_dev.model"

    print("Loading model: {}".format(model_path))
    try:
        model = Word2Vec.load(model_path)
    except FileNotFoundError:
        print("Error: Model file '{}' not found.".format(model_path))
        print("Please make sure the file is in the current directory.")
        return
    except Exception as e:
        print("Error loading model: {}".format(e))
        return

    vocab_size = len(model.wv.key_to_index)
    vector_size = model.vector_size
    print("Model loaded successfully. Vocabulary size: {}, Vector dimension: {}".format(vocab_size, vector_size))
    print()

    print("Usage:")
    print("- Enter one word (e.g., good) to see most similar words")
    print("- Enter two words (e.g., good great) to compute similarity")
    print("- Type 'vector WORD' (e.g., vector amazing) to show first 10 dimensions of the vector")
    print("- Type 'quit' or 'exit' to exit")
    print()

    while True:
        try:
            user_input = input("Enter word(s) or command: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if user_input.lower() in ['quit', 'exit']:
            print("Exiting...")
            break

        if not user_input:
            continue

        # Handle 'vector' command
        if user_input.startswith("vector "):
            word = user_input[7:].strip()
            if word in model.wv:
                vec = model.wv[word]
                print("Vector for '{}':\n{}\n".format(word, vec[:10]))
            else:
                print("Word '{}' not in vocabulary.\n".format(word))
            continue

        words = user_input.split()

        if len(words) == 1:
            word = words[0]
            if word in model.wv:
                print("Most similar words to '{}':".format(word))
                similar = model.wv.most_similar(word, topn=10)
                for i, (w, score) in enumerate(similar, 1):
                    print("  {}. {:<15} ({:.4f})".format(i, w, score))
                print()
            else:
                print("Word '{}' not in vocabulary.\n".format(word))

        elif len(words) == 2:
            w1, w2 = words
            if w1 in model.wv and w2 in model.wv:
                sim = model.wv.similarity(w1, w2)
                print("Similarity between '{}' and '{}': {:.4f}\n".format(w1, w2, sim))
            else:
                missing = [w for w in [w1, w2] if w not in model.wv]
                print("The following words are not in vocabulary: {}\n".format(", ".join(missing)))

        else:
            print("Please enter one word, two words, or use 'vector WORD'.\n")


if __name__ == "__main__":
    main()