package scratch;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.lang.instrument.ClassDefinition;
import java.lang.instrument.Instrumentation;

public class LiveAgent {
    public static void agentmain(String args, Instrumentation inst) {
        File logFile = new File("live_hotreload.log");
        try (FileWriter log = new FileWriter(logFile, true)) {
            log.write("[LiveAgent] Running with args: " + args + "\n");
            String[] parts = args.split("\\|");
            if (parts.length < 2) {
                log.write("[LiveAgent] Error: parts length < 2\n");
                return;
            }
            String className = parts[0];
            String filePath = parts[1];
            File f = new File(filePath);
            if (!f.exists()) {
                log.write("[LiveAgent] Error: file does not exist: " + filePath + "\n");
                return;
            }
            byte[] bytes = new byte[(int) f.length()];
            try (FileInputStream in = new FileInputStream(f)) {
                int read = in.read(bytes);
                log.write("[LiveAgent] Read " + read + " bytes for " + className + "\n");
            }

            Class<?> targetClass = null;
            for (Class<?> c : inst.getAllLoadedClasses()) {
                if (c.getName().equals(className)) {
                    targetClass = c;
                    break;
                }
            }

            if (targetClass != null) {
                inst.redefineClasses(new ClassDefinition(targetClass, bytes));
                log.write("[LiveAgent] SUCCESS: " + className + " redefined successfully!\n");
            } else {
                log.write("[LiveAgent] FAILED: Class " + className + " not found among " + inst.getAllLoadedClasses().length + " classes\n");
            }
        } catch (Throwable t) {
            try (FileWriter errLog = new FileWriter(logFile, true)) {
                errLog.write("[LiveAgent] EXCEPTION: " + t + "\n");
                for (StackTraceElement el : t.getStackTrace()) {
                    errLog.write("  at " + el + "\n");
                }
            } catch (Exception ignored) {}
        }
    }
}
