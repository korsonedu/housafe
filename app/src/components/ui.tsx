import { View, Text, ViewProps } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { colors, space, radius, font, shadow } from "../theme";

export function Screen({ children }: ViewProps) {
  return (
    <SafeAreaView style={{ flex:1, backgroundColor: colors.bg }}>
      <View style={{ flex:1, paddingHorizontal: space.xl }}>{children}</View>
    </SafeAreaView>
  );
}

export function Card({ children, tone="normal" }: { children: any; tone?: "normal"|"attention"|"alert" }) {
  const bar = { normal: colors.primary, attention: colors.attention, alert: colors.alert }[tone];
  return (
    <View style={{ backgroundColor: colors.surface, borderRadius: radius.lg, padding: space.lg,
      marginBottom: space.md, borderLeftWidth:4, borderLeftColor: bar, ...shadow.card }}>
      {children}
    </View>
  );
}

export function StatusBadge({ online }: { online: boolean }) {
  const c = online ? colors.primary : colors.textMuted;
  const soft = online ? colors.primarySoft : colors.border;
  return (
    <View style={{ flexDirection:"row", alignItems:"center", gap: space.xs, alignSelf:"flex-start",
      backgroundColor: soft, paddingHorizontal: space.md, paddingVertical: space.xs, borderRadius: radius.pill }}>
      <View style={{ width:8, height:8, borderRadius:4, backgroundColor:c }} />
      <Text style={{ color:c, fontSize: font.small, fontWeight:"600" }}>{online ? "在线" : "离线"}</Text>
    </View>
  );
}

export function VitalStat({ icon, label, value }: { icon: any; label: string; value: string }) {
  return (
    <View style={{ flex:1, alignItems:"center", gap: space.xs }}>
      <Ionicons name={icon} size={22} color={colors.primary} />
      <Text style={{ fontSize: font.h2, fontWeight:"700", color: colors.text }}>{value}</Text>
      <Text style={{ fontSize: font.small, color: colors.textMuted }}>{label}</Text>
    </View>
  );
}

export function SectionHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <View style={{ marginTop: space.lg, marginBottom: space.md }}>
      <Text style={{ fontSize: font.h1, fontWeight:"700", color: colors.text }}>{title}</Text>
      {subtitle ? <Text style={{ fontSize: font.body, color: colors.textMuted, marginTop: space.xs }}>{subtitle}</Text> : null}
    </View>
  );
}
