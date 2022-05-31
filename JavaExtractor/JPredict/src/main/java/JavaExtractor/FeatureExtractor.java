package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import JavaExtractor.Common.MethodContent;
import JavaExtractor.FeaturesEntities.ProgramFeatures;
import JavaExtractor.FeaturesEntities.Property;
import JavaExtractor.Visitors.FunctionVisitor;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.printer.DotPrinter;
import com.github.javaparser.ast.comments.LineComment;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import java.util.StringJoiner;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.io.PrintWriter;
import java.io.FileWriter;

@SuppressWarnings("StringEquality")
class FeatureExtractor {
  private final static String upSymbol = "|";
  private final static String downSymbol = "|";
  private static final Set<String> s_ParentTypeToAddChildId = Stream
      .of("AssignExpr", "ArrayAccessExpr", "FieldAccessExpr", "MethodCallExpr")
      .collect(Collectors.toCollection(HashSet::new));
  private final CommandLineValues m_CommandLineValues;

  public FeatureExtractor(CommandLineValues commandLineValues) {
    this.m_CommandLineValues = commandLineValues;
  }

  private static ArrayList<Node> getTreeStack(Node node) {
    ArrayList<Node> upStack = new ArrayList<>();
    Node current = node;
    while (current != null) {
      upStack.add(current);
      try {
        current = current.getParentNode().get();
      } catch (Exception e) {
        break;
      }

    }
    return upStack;
  }

  public ArrayList<ProgramFeatures> extractFeatures(String code) {
    CompilationUnit m_CompilationUnit = parseFileWithRetries(code);

    // generates a dot file that can be converted into
    // and AST with Graphviz
    if (m_CommandLineValues.GenerateAST) {
      DotPrinter printer = new DotPrinter(true);
      try (FileWriter fileWriter = new FileWriter("ast.dot");
          PrintWriter printWriter = new PrintWriter(fileWriter)) {
        printWriter.print(printer.output(m_CompilationUnit));
      } catch (Exception e) {
        System.out.println(e);
      }
    }

    FunctionVisitor functionVisitor = new FunctionVisitor(m_CommandLineValues);

    functionVisitor.visit(m_CompilationUnit, null);

    ArrayList<MethodContent> methods = functionVisitor.getMethodContents();

    return generatePathFeatures(methods);
  }

  private CompilationUnit parseFileWithRetries(String code) {
    final String classPrefix = "public class Test {";
    final String classSuffix = "}";
    final String methodPrefix = "SomeUnknownReturnType f() {";
    final String methodSuffix = "return noSuchReturnValue; }";

    String content = code;
    CompilationUnit parsed;
    try {
      parsed = JavaParser.parse(content);
    } catch (ParseProblemException e1) {
      // Wrap with a class and method
      try {
        content = classPrefix + methodPrefix + code + methodSuffix + classSuffix;
        parsed = JavaParser.parse(content);
      } catch (ParseProblemException e2) {
        // Wrap with a class only
        // try {
          content = classPrefix + code + classSuffix;
          parsed = JavaParser.parse(content);
        // } catch (ParseProblemException e3) { // Idea for future: dont catch anything to let it fail and add nothing! 
        //   content = "";
        //   parsed = JavaParser.parse(content);
        // }
      }
    }
    return parsed;
  }

  private ArrayList<ProgramFeatures> generatePathFeatures(ArrayList<MethodContent> methods) {
    ArrayList<ProgramFeatures> methodsFeatures = new ArrayList<>();
    for (MethodContent content : methods) {
      ProgramFeatures singleMethodFeatures = generatePathFeaturesForFunction(content);
      if (!singleMethodFeatures.isEmpty()) {
        methodsFeatures.add(singleMethodFeatures);
      }
    }
    return methodsFeatures;
  }

  private ProgramFeatures generatePathFeaturesForFunction(MethodContent methodContent) {
    ArrayList<Node> functionLeaves = methodContent.getLeaves();
    ProgramFeatures programFeatures = new ProgramFeatures(methodContent.getName());

    for (int i = 0; i < functionLeaves.size(); i++) {
      for (int j = i + 1; j < functionLeaves.size(); j++) {
        String separator = Common.EmptyString;

        String path = generatePath(functionLeaves.get(i), functionLeaves.get(j), separator);
        if (path != Common.EmptyString) {
          Property source = functionLeaves.get(i).getData(Common.PropertyKey);
          Property target = functionLeaves.get(j).getData(Common.PropertyKey);
          programFeatures.addFeature(source, path, target);
        }
      }
    }
    return programFeatures;
  }

  private String generatePath(Node source, Node target, String separator) {

    StringJoiner stringBuilder = new StringJoiner(separator);
    ArrayList<Node> sourceStack = getTreeStack(source);
    ArrayList<Node> targetStack = getTreeStack(target);

    int commonPrefix = 0;
    int currentSourceAncestorIndex = sourceStack.size() - 1;
    int currentTargetAncestorIndex = targetStack.size() - 1;
    while (currentSourceAncestorIndex >= 0 && currentTargetAncestorIndex >= 0
        && sourceStack.get(currentSourceAncestorIndex) == targetStack.get(currentTargetAncestorIndex)) {
      commonPrefix++;
      currentSourceAncestorIndex--;
      currentTargetAncestorIndex--;
    }

    int pathLength = sourceStack.size() + targetStack.size() - 2 * commonPrefix;
    if (pathLength > m_CommandLineValues.MaxPathLength) {
      return Common.EmptyString;
    }

    if (currentSourceAncestorIndex >= 0 && currentTargetAncestorIndex >= 0) {
      int pathWidth = targetStack.get(currentTargetAncestorIndex).getData(Common.ChildId)
          - sourceStack.get(currentSourceAncestorIndex).getData(Common.ChildId);
      if (pathWidth > m_CommandLineValues.MaxPathWidth) {
        return Common.EmptyString;
      }
    }

    for (int i = 0; i < sourceStack.size() - commonPrefix; i++) {
      Node currentNode = sourceStack.get(i);
      String childId = Common.EmptyString;
      String parentRawType = currentNode.getParentNode().get().getData(Common.PropertyKey).getRawType();
      if (i == 0 || s_ParentTypeToAddChildId.contains(parentRawType)) {
        childId = saturateChildId(currentNode.getData(Common.ChildId))
            .toString();
      }
      stringBuilder.add(String.format("%s%s%s",
          currentNode.getData(Common.PropertyKey).getType(true), childId, upSymbol));
    }

    Node commonNode = sourceStack.get(sourceStack.size() - commonPrefix);
    String commonNodeChildId = Common.EmptyString;
    Property parentNodeProperty = commonNode.getParentNode().get().getData(Common.PropertyKey);
    String commonNodeParentRawType = Common.EmptyString;
    if (parentNodeProperty != null) {
      commonNodeParentRawType = parentNodeProperty.getRawType();
    }
    if (s_ParentTypeToAddChildId.contains(commonNodeParentRawType)) {
      commonNodeChildId = saturateChildId(commonNode.getData(Common.ChildId))
          .toString();
    }
    stringBuilder.add(String.format("%s%s",
        commonNode.getData(Common.PropertyKey).getType(true), commonNodeChildId));

    for (int i = targetStack.size() - commonPrefix - 1; i >= 0; i--) {
      Node currentNode = targetStack.get(i);
      String childId = Common.EmptyString;
      if (i == 0 || s_ParentTypeToAddChildId.contains(currentNode.getData(Common.PropertyKey).getRawType())) {
        childId = saturateChildId(currentNode.getData(Common.ChildId))
            .toString();
      }
      stringBuilder.add(String.format("%s%s%s", downSymbol,
          currentNode.getData(Common.PropertyKey).getType(true), childId));
    }

    return stringBuilder.toString();
  }

  private Integer saturateChildId(int childId) {
    return Math.min(childId, m_CommandLineValues.MaxChildId);
  }
}
