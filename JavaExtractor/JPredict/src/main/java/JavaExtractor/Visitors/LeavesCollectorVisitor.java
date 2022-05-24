package JavaExtractor.Visitors;

import JavaExtractor.Common.Common;
import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.FeaturesEntities.Property;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.expr.NullLiteralExpr;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.visitor.TreeVisitor;
import com.github.javaparser.ast.comments.JavadocComment;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Arrays;

public class LeavesCollectorVisitor extends TreeVisitor {
  private final ArrayList<Node> m_Leaves = new ArrayList<>();
  private final CommandLineValues m_CommandLineValues;
  public static final ArrayList<String> stopWords = new ArrayList<>(Arrays.asList("I", "a", "about", "above", "after",
      "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
      "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did",
      "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
      "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
      "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if",
      "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
      "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
      "ours 	ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
      "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
      "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
      "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
      "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
      "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
      "your", "yours", "yourself", "yourselves"));

  public LeavesCollectorVisitor(CommandLineValues commandLineValues) {
    this.m_CommandLineValues = commandLineValues;
  }

  @Override
  public void process(Node node) {

    if (m_CommandLineValues.InlineComments) {
      // If node has comment add it to leaves.
      // Check if it is not javadoc or contains code.
      Optional<Comment> com = node.getComment();
      if (com.isPresent() && !(com.get() instanceof JavadocComment) && !containsCode(com.get().getContent())) {
        String updatedComment = removeStopWords(com.get().getContent());
        m_Leaves.add(node.setComment(com.get().setContent(updatedComment)));
      }
    }

    if (node instanceof Comment) {
      return;
    }
    boolean isLeaf = false;
    boolean isGenericParent = isGenericParent(node);
    if (hasNoChildren(node) && isNotComment(node)) {
      if (!node.toString().isEmpty() && (!"null".equals(node.toString()) || (node instanceof NullLiteralExpr))) {
        m_Leaves.add(node);
        isLeaf = true;
      }
    }

    int childId = getChildId(node);
    node.setData(Common.ChildId, childId);
    Property property = new Property(node, isLeaf, isGenericParent);
    node.setData(Common.PropertyKey, property);
  }

  private boolean containsCode(String comment) {
    return ("//" + comment).matches("^\\s*.*;\\s*$");
  }

  private String removeStopWords(String comment) {
    String[] words = comment.split(" ");
    ArrayList<String> wordsList = new ArrayList<String>();

    for (String word : words) {
      if (!stopWords.contains(word)) {
        wordsList.add(word);
      }
    }
    return String.join(" ", wordsList);
  }

  private boolean isGenericParent(Node node) {
    return (node instanceof ClassOrInterfaceType)
        && ((ClassOrInterfaceType) node).getTypeArguments().isPresent()
        && ((ClassOrInterfaceType) node).getTypeArguments().get().size() > 0;
  }

  private boolean hasNoChildren(Node node) {
    return node.getChildNodes().size() == 0;

  }

  private boolean isNotComment(Node node) {
    return !(node instanceof Comment) && !(node instanceof Statement);
  }

  public ArrayList<Node> getLeaves() {
    return m_Leaves;
  }

  private int getChildId(Node node) {
    Node parent = node.getParentNode().get();
    List<Node> parentsChildren = parent.getChildNodes();
    int childId = 0;
    for (Node child : parentsChildren) {
      if (child.getRange().equals(node.getRange())) {
        return childId;
      }
      childId++;
    }
    return childId;
  }
}
