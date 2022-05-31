package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import JavaExtractor.FeaturesEntities.ProgramFeatures;
import org.apache.commons.lang3.StringUtils;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.lang.StringBuilder;

import org.json.JSONObject;
import org.json.JSONArray;

class ExtractFeaturesTask implements Callable<Void> {
  private final CommandLineValues m_CommandLineValues;
  private final String fileLine;

  public ExtractFeaturesTask(CommandLineValues commandLineValues, String line) {
    m_CommandLineValues = commandLineValues;
    this.fileLine = line;
  }

  @Override
  public Void call() {
    processFile();
    return null;
  }

  public void processFile() {
    ArrayList<ProgramFeatures> features;
    try {
      features = extractSingleFile();
    } catch (IOException e) {
      e.printStackTrace();
      return;
    }
    if (features == null) {
      return;
    }

    String toPrint = featuresToString(features);
    if (toPrint.length() > 0) {
      System.out.println(toPrint);
    }
  }

  private ArrayList<ProgramFeatures> extractSingleFile() throws IOException {

    // Modified to only extract strings, not files;
    String code = collectJson(fileLine);

    FeatureExtractor featureExtractor = new FeatureExtractor(m_CommandLineValues);

    return featureExtractor.extractFeatures(code);
  }

  private String collectJson(String json) {
    JSONObject jo = new JSONObject(json);

    StringBuilder full_entry = new StringBuilder();
    StringBuilder javaDoc = new StringBuilder().append("/**");
    String doc = (String) jo.get("docstring");

    String summary = doc.lines()
        .filter(l -> !l.contains("=") && !l.contains("-") && !l.startsWith("<") && !l.startsWith("(")
            && !l.startsWith("@") && l.split("[\\s+]").length >= 2)
        .findFirst().orElseThrow();

    javaDoc.append(summary + "\n");
    javaDoc.append("*/\n");

    full_entry.append(javaDoc);
    full_entry.append(jo.get("original_string") + "\n");

    return full_entry.toString();
  }

  public String featuresToString(ArrayList<ProgramFeatures> features) {
    if (features == null || features.isEmpty()) {
      return Common.EmptyString;
    }

    List<String> methodsOutputs = new ArrayList<>();

    for (ProgramFeatures singleMethodFeatures : features) {
      StringBuilder builder = new StringBuilder();

      String toPrint = singleMethodFeatures.toString();
      if (m_CommandLineValues.PrettyPrint) {
        toPrint = toPrint.replace(" ", "\n\t");
      }
      builder.append(toPrint);

      methodsOutputs.add(builder.toString());

    }
    return StringUtils.join(methodsOutputs, "\n");
  }
}
