import { useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Screen } from "./components/ui";
import { colors, space, radius, font, shadow } from "./theme";
import { login } from "./api";

interface Props {
  onLogin: (token: string) => void;
}

export default function LoginScreen({ onLogin }: Props) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async () => {
    setError("");
    if (!username.trim() || !password.trim()) {
      setError("请输入用户名和密码");
      return;
    }
    setLoading(true);
    try {
      const token = await login(username.trim(), password);
      onLogin(token);
    } catch (e: any) {
      setError(e.message || "登录失败，请重试");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Screen>
      <KeyboardAvoidingView
        style={{ flex: 1, justifyContent: "center" }}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
      >
        {/* Logo */}
        <View style={{ alignItems: "center", marginBottom: space.xxl }}>
          <View
            style={{
              width: 80,
              height: 80,
              borderRadius: 40,
              backgroundColor: colors.primarySoft,
              alignItems: "center",
              justifyContent: "center",
              marginBottom: space.lg,
            }}
          >
            <Ionicons name="shield-checkmark" size={40} color={colors.primary} />
          </View>
          <Text style={{ fontSize: font.h1, fontWeight: "700", color: colors.text }}>
            家安
          </Text>
          <Text
            style={{
              fontSize: font.body,
              color: colors.textMuted,
              marginTop: space.xs,
            }}
          >
            随时守护家人安全
          </Text>
        </View>

        {/* Form */}
        <View
          style={{
            backgroundColor: colors.surface,
            borderRadius: radius.lg,
            padding: space.xl,
            ...shadow.card,
          }}
        >
          <View style={{ marginBottom: space.lg }}>
            <Text
              style={{
                fontSize: font.small,
                color: colors.textMuted,
                marginBottom: space.xs,
              }}
            >
              用户名
            </Text>
            <TextInput
              style={{
                backgroundColor: colors.bg,
                borderRadius: radius.md,
                padding: space.md,
                fontSize: font.body,
                color: colors.text,
                borderWidth: 1,
                borderColor: colors.border,
              }}
              placeholder="请输入用户名"
              placeholderTextColor={colors.textMuted}
              autoCapitalize="none"
              value={username}
              onChangeText={setUsername}
              editable={!loading}
            />
          </View>

          <View style={{ marginBottom: space.lg }}>
            <Text
              style={{
                fontSize: font.small,
                color: colors.textMuted,
                marginBottom: space.xs,
              }}
            >
              密码
            </Text>
            <TextInput
              style={{
                backgroundColor: colors.bg,
                borderRadius: radius.md,
                padding: space.md,
                fontSize: font.body,
                color: colors.text,
                borderWidth: 1,
                borderColor: colors.border,
              }}
              placeholder="请输入密码"
              placeholderTextColor={colors.textMuted}
              secureTextEntry
              value={password}
              onChangeText={setPassword}
              editable={!loading}
              onSubmitEditing={handleLogin}
            />
          </View>

          {error !== "" && (
            <Text
              style={{
                color: colors.alert,
                fontSize: font.small,
                marginBottom: space.md,
                textAlign: "center",
              }}
            >
              {error}
            </Text>
          )}

          <TouchableOpacity
            style={{
              backgroundColor: colors.primary,
              borderRadius: radius.md,
              padding: space.md,
              alignItems: "center",
              opacity: loading ? 0.7 : 1,
            }}
            onPress={handleLogin}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color={colors.surface} />
            ) : (
              <Text
                style={{
                  color: colors.surface,
                  fontSize: font.body,
                  fontWeight: "600",
                }}
              >
                登录
              </Text>
            )}
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </Screen>
  );
}
