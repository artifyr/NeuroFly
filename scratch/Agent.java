package scratch;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.lang.instrument.ClassDefinition;
import java.lang.instrument.Instrumentation;

public class Agent {
    public static void agentmain(String args, Instrumentation inst) {
        try (FileWriter log = new FileWriter("hotreload.log", true)) {
            log.write("[HotReload] Args received: " + args + "\n");
            String[] parts = args.split("\\|");
            if (parts.length < 2) {
                log.write("[HotReload] Error: parts length < 2\n");
                return;
            }
            String className = parts[0];
            String filePath = parts[1];
            File f = new File(filePath);
            if (!f.exists()) {
                log.write("[HotReload] Error: file does not exist: " + filePath + "\n");
                return;
            }
            byte[] bytes = new byte[(int) f.length()];
            try (FileInputStream in = new FileInputStream(f)) {
                int read = in.read(bytes);
                log.write("[HotReload] Read " + read + " bytes\n");
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
                log.write("[HotReload] SUCCESS: " + className + " redefined successfully!\n");
            } else {
                log.write("[HotReload] FAILED: Class " + className + " not found in loaded classes!\n");
            }
        } catch (Throwable t) {
            try (FileWriter errLog = new FileWriter("hotreload.log", true)) {
                errLog.write("[HotReload] EXCEPTION: " + t.toString() + "\n");
                for (StackTraceElement elem : t.getStackTrace()) {
                    errLog.write("  at " + elem.toString() + "\n");
                }
            } catch (Exception ignored) {}
        }
    }
}
