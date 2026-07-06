# Add project specific ProGuard rules here.
# minifyEnabled is currently false (see app/build.gradle), so this file is
# inert until you turn minification on for release builds. If you do:
#  - Keep JNI-called methods so R8 doesn't strip them:
-keepclasseswithmembernames class * {
    native <methods>;
}
-keep class com.medrag.offline.llm.LlamaBridge { *; }
