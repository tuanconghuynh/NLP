from src.summarizer import summarize_text

text = """
Artificial Intelligence is transforming the world.
It is being used in healthcare, education, and transportation.
Machine learning is an important part of artificial intelligence.
It allows computers to learn from data and make predictions.
AI can improve productivity and reduce human effort.
Many companies are investing in artificial intelligence.
AI technologies are becoming increasingly popular.
However, AI also raises concerns about privacy and security.
Researchers are working to develop safer AI systems.
Artificial intelligence will continue to influence society.
"""

summary = summarize_text(text, ratio=0.3)

print("=" * 60)
print("ORIGINAL TEXT")
print("=" * 60)
print(text)

print("\n" + "=" * 60)
print("GENERATED SUMMARY")
print("=" * 60)
print(summary)