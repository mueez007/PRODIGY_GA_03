import spacy
import re
import markovify
import nltk
from nltk.corpus import gutenberg
import warnings
import os

warnings.filterwarnings('ignore')

# Download required NLTK data
def setup_nltk():
    print("Setting up NLTK data...")
    nltk.download('gutenberg', quiet=True)
    nltk.download('punkt', quiet=True)
    print("NLTK setup complete!")

# Text cleaning function
def text_cleaner(text):
    text = re.sub(r'--', ' ', text)
    text = re.sub('[[].*?[]]', '', text)
    text = re.sub(r'(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b', '', text)
    text = ' '.join(text.split())
    return text

# POSified Text class for better grammar
class POSifiedText(markovify.Text):
    def word_split(self, sentence):
        return ['::'.join((word.orth_, word.pos_)) for word in nlp(sentence)]
    
    def word_join(self, words):
        sentence = ' '.join(word.split('::')[0] for word in words)
        return sentence

def generate_guaranteed_sentence(generator, max_tries=100):
    """Generate a sentence with multiple fallback methods"""
    # First try: Normal generation with high tries
    sentence = generator.make_sentence(tries=max_tries)
    if sentence:
        return sentence
    
    # Second try: Disable novelty check
    sentence = generator.make_sentence(test_output=False)
    if sentence:
        return sentence
    
    # Third try: Generate short sentence
    sentence = generator.make_short_sentence(140)
    if sentence:
        return sentence
    
    # Final fallback: Force generation
    return generator.make_sentence(test_output=False, tries=200)

def main():
    # Setup
    setup_nltk()
    
    # Load spaCy model
    print("Loading spaCy model...")
    global nlp
    nlp = spacy.load('en_core_web_sm')
    
    # Load and prepare Shakespeare texts
    print("Loading Shakespeare texts...")
    hamlet = gutenberg.raw('shakespeare-hamlet.txt')
    macbeth = gutenberg.raw('shakespeare-macbeth.txt')
    caesar = gutenberg.raw('shakespeare-caesar.txt')
    
    # Clean the texts
    print("Cleaning texts...")
    hamlet = re.sub(r'Chapter \d+', '', hamlet)
    macbeth = re.sub(r'Chapter \d+', '', macbeth)
    caesar = re.sub(r'Chapter \d+', '', caesar)
    
    hamlet = text_cleaner(hamlet)
    macbeth = text_cleaner(macbeth)
    caesar = text_cleaner(caesar)
    
    # Parse with spaCy
    print("Parsing with spaCy...")
    hamlet_doc = nlp(hamlet)
    macbeth_doc = nlp(macbeth)
    caesar_doc = nlp(caesar)
    
    # Create sentences
    hamlet_sents = ' '.join([sent.text for sent in hamlet_doc.sents if len(sent.text) > 1])
    macbeth_sents = ' '.join([sent.text for sent in macbeth_doc.sents if len(sent.text) > 1])
    caesar_sents = ' '.join([sent.text for sent in caesar_doc.sents if len(sent.text) > 1])
    
    shakespeare_sents = hamlet_sents + macbeth_sents + caesar_sents
    
    # Create basic Markov generator
    print("Training basic Markov model...")
    generator_1 = markovify.Text(shakespeare_sents, state_size=3)
    
    # Generate basic text
    basic_output = []
    basic_output.append("=== BASIC MARKOV GENERATION ===")
    basic_output.append("Three random sentences:")
    for i in range(3):
        sentence = generate_guaranteed_sentence(generator_1)
        basic_output.append(f"{i+1}. {sentence}")
    
    basic_output.append("\nThree short sentences (max 100 chars):")
    for i in range(3):
        sentence = generator_1.make_short_sentence(max_chars=100, tries=100)
        if not sentence:
            sentence = generate_guaranteed_sentence(generator_1)
        basic_output.append(f"{i+1}. {sentence}")
    
    # Create advanced generator
    print("Training POS-aware Markov model...")
    generator_2 = POSifiedText(shakespeare_sents, state_size=3)
    
    # Generate advanced text
    advanced_output = []
    advanced_output.append("=== ADVANCED POS-AWARE GENERATION ===")
    advanced_output.append("Five random sentences:")
    for i in range(5):
        sentence = generate_guaranteed_sentence(generator_2)
        advanced_output.append(f"{i+1}. {sentence}")
    
    advanced_output.append("\nFive short sentences (max 100 chars):")
    for i in range(5):
        sentence = generator_2.make_short_sentence(max_chars=100, tries=100)
        if not sentence:
            sentence = generate_guaranteed_sentence(generator_2)
        advanced_output.append(f"{i+1}. {sentence}")
    
    # Print to console
    print("\n" + "\n".join(basic_output))
    print("\n" + "\n".join(advanced_output))
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save to files
    with open('output/basic_generated.txt', 'w') as f:
        f.write("\n".join(basic_output))
    
    with open('output/advanced_generated.txt', 'w') as f:
        f.write("\n".join(advanced_output))
    
    print("\n✅ Generated texts saved to output/ directory!")
    print("📁 Files created:")
    print("   - output/basic_generated.txt")
    print("   - output/advanced_generated.txt")
    print("\n🎉 Task 3 Completed Successfully!")

if __name__ == "__main__":
    main()