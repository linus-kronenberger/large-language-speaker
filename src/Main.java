import java.util.List;

public class Main {
  static void main() {
    Main main = new Main();
    String prompt = "Hi, I am Linus der";
    String result = main.generateText(prompt);
    System.out.println("User: " + prompt);
    System.out.println("------------------------------");
    System.out.println("Assistant: " + result);
  }

  public String generateText(String prompt) {
    PositionalTokenizer tokenizer = new PositionalTokenizer();
    List<List<Float>> posEmbeddings = tokenizer.tokenize(prompt);
    // MultiSelfAttentionProcessor selfAttentionProcessor = new MultiSelfAttentionProcessor();
    // List<List<Float>> semEmbeddings = selfAttentionProcessor.embedd(List<List<Float>>));
    // FeedforwardProcessor feedforwarder = new FeedforwardProcessor();
    // List<List<Float>> meanEmbeddings = feedforwarder.forward(semEmbeddings);
    // WordPredictor predictor = new WordPredictor();
    // String nextWord = predictor.predict();
    return "";
  }
}
