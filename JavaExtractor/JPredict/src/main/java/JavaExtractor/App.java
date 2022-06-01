package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import org.kohsuke.args4j.CmdLineException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

public class App {
  private static CommandLineValues s_CommandLineValues;

  public static void main(String[] args) {
    try {
      s_CommandLineValues = new CommandLineValues(args);
    } catch (CmdLineException e) {
      e.printStackTrace();
      return;
    }

    if (s_CommandLineValues.File != null) {
      String code;
      try {
        code = new String(Files.readAllBytes(s_CommandLineValues.File.toPath()));
      } catch (IOException e) {
        e.printStackTrace();
        code = Common.EmptyString;
      }

      ExtractFeaturesTask extractFeaturesTask = new ExtractFeaturesTask(s_CommandLineValues, code);
      extractFeaturesTask.processFile();
    } else if (s_CommandLineValues.Dir != null) {
      extractDir();
    }
  }

  private static void extractDir() {
    ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(s_CommandLineValues.NumThreads);
    LinkedList<ExtractFeaturesTask> tasks = new LinkedList<>();
    try {
      Files.walk(Paths.get(s_CommandLineValues.Dir)).filter(Files::isRegularFile)
          .filter(p -> p.toString().toLowerCase().endsWith(".jsonl")).forEach(f -> {
            List<String> code;
            // create file from path (see extractSingleFile on how)
            try {
              code = Files.readAllLines(f);
            } catch (IOException e) {
              e.printStackTrace();
              code = new ArrayList<>();
            }
            // For each line in file create a new tasks
            code.forEach(l -> {
              ExtractFeaturesTask task = new ExtractFeaturesTask(s_CommandLineValues, l);
              tasks.add(task);
            });

          });
    } catch (IOException e) {
      e.printStackTrace();
      return;
    }
    List<Future<Void>> tasksResults = null;
    try {
      tasksResults = executor.invokeAll(tasks);
    } catch (InterruptedException e) {
      e.printStackTrace();
    } finally {
      executor.shutdown();
    }
    tasksResults.forEach(f -> {
      try {
        f.get();
      } catch (InterruptedException | ExecutionException e) {
        // e.printStackTrace();
      }
    });
  }
}
