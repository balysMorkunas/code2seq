package JavaExtractor.Common;

import JavaExtractor.FeaturesEntities.Property;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.DataKey;

import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public final class Common {
    public static final DataKey<Property> PropertyKey = new DataKey<Property>() {
    };
    public static final DataKey<Integer> ChildId = new DataKey<Integer>() {
    };
    public static final String EmptyString = "";

    public static final String MethodDeclaration = "MethodDeclaration";
    public static final String NameExpr = "NameExpr";
    public static final String BlankWord = "BLANK";

    public static final int c_MaxLabelLength = 50;
    public static final String methodName = "METHOD_NAME";
    public static final String internalSeparator = "|";

    public static String normalizeName(String original, String defaultString) {
        original = original.toLowerCase().replaceAll("\\\\n", "") // escaped new
                // lines
                .replaceAll("//s+", "") // whitespaces
                .replaceAll("[\"',]", "") // quotes, apostrophies, commas
                .replaceAll("\\P{Print}", ""); // unicode weird characters
        String stripped = original.replaceAll("[^A-Za-z]", "");
        if (stripped.length() == 0) {
            String carefulStripped = original.replaceAll(" ", "_");
            if (carefulStripped.length() == 0) {
                return defaultString;
            } else {
                return carefulStripped;
            }
        } else {
            return stripped;
        }
    }

    public static boolean isMethod(Node node, String type) {
        Property parentProperty = node.getParentNode().get().getData(Common.PropertyKey);
        if (parentProperty == null) {
            return false;
        }

        String parentType = parentProperty.getType();
        return Common.NameExpr.equals(type) && Common.MethodDeclaration.equals(parentType);
    }

    public static ArrayList<String> splitToSubtokens(String str1) {
        String str2 = str1.replace("|", " ");
        String str3 = str2.trim();
        return Stream.of(str3.split("(?<=[a-z])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\\s+"))
                .filter(s -> s.length() > 0).map(s -> Common.normalizeName(s, Common.EmptyString))
                .filter(s -> s.length() > 0).map(s -> parseComment(s))
                .filter(s -> s.length() > 0).collect(Collectors.toCollection(ArrayList::new));
    }

    public static String parseComment(String comment) {
        String parsed = comment.replaceAll("[^a-zA-Z0-9]", "");
        return parsed;
  }
}
