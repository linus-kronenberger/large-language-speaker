import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class PositionalTokenizer {

  private final int EMBEDDING_DIMENSIONS = 64;
  private final String DATA_FILE_PATH = "./dataset/speeches.csv";

  public List<List<Float>> tokenize(String sentence) {
    Map<String, Integer> vocabulary = buildVocabulary();
    List<List<Float>> tokenPosEmbeddings = new ArrayList<>(List.of());
    List<String> wordList = List.of(sentence.split(" |(\\n)"));
    wordList.forEach(word -> {
      int token = wordEncode(word, vocabulary);
      List<Float> tokenEmbedding = new ArrayList<>(List.of());
      float tokenFloat = (float) token;
      System.out.print("[");
      for (int i = 0; i < EMBEDDING_DIMENSIONS; i++) {
        tokenEmbedding.add(tokenFloat);
        System.out.print(tokenFloat + " ");
      }
      System.out.println("]");
      System.out.println();
      tokenPosEmbeddings.add(tokenEmbedding);
    });
    return tokenPosEmbeddings;
  }

  public int wordEncode(String word, Map<String, Integer> vocabulary) {
    if(!(vocabulary.get(word) == null)) {
      return vocabulary.get(word);
    } else {
      return vocabulary.get("NULL");
    }
  }

  public Map<String, Integer > buildVocabulary() {
    System.out.println("Building Vocabulary..");
    Map<String, Integer> vocabulary = new HashMap<>();
    String line = "";
    int vocabLimit = 10;
    int count = 0;
    vocabulary.put("NULL", 0);
    AtomicInteger wordIndex = new AtomicInteger(1);
    try (BufferedReader br = new BufferedReader(new FileReader(DATA_FILE_PATH))) {
      while ((line = br.readLine()) != null) {
        System.out.println(line);
        List<String> vocabWords = List.of(line.split(" "));
        vocabWords.forEach(word -> {
          if(!vocabulary.containsKey(word)) {
            vocabulary.put(word, wordIndex.get());
            System.out.println("Added token to vocabulary: " + wordIndex + "   " + word);
            wordIndex.getAndIncrement();
          }
        });
        count ++;
        if(count == vocabLimit) {
          break;
        }
      }
    } catch (IOException e) {
      e.printStackTrace();
    }

    return vocabulary;
  }
}
