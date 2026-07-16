import { useState, useEffect, useRef, useCallback } from "react";
import {
  View,
  Text,
  ActivityIndicator,
  ScrollView,
  TouchableOpacity,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Screen, Card, StatusBadge, VitalStat, SectionHeader } from "./components/ui";
import { colors, space, radius, font } from "./theme";
import { getFamilies, getToday, WS_BASE } from "./api";

interface Props {
  token: string;
  onLogout: () => void;
}

const POSTURE_CN: Record<string, string> = {
  stand: "站立",
  sit: "坐着",
  lie: "躺卧",
  walk: "走动",
  fall: "跌倒",
};

interface RoomData {
  room_name: string;
  device_online: boolean;
  posture: string | null;
  heart_rate: number | null;
  breath_rate: number | null;
}

type LoadState = "loading" | "empty" | "error" | "ok";

export default function TodayScreen({ token, onLogout }: Props) {
  const [familyId, setFamilyId] = useState<number | null>(null);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [errorMsg, setErrorMsg] = useState("");
  const [rooms, setRooms] = useState<RoomData[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const mergeUpdate = useCallback((update: any) => {
    setRooms((prev) => {
      const next = [...prev];
      const idx = next.findIndex((r) => r.room_name === update.room_name);
      const merged: RoomData = {
        room_name: update.room_name ?? "",
        device_online:
          update.device_online !== undefined
            ? update.device_online
            : idx >= 0
            ? next[idx].device_online
            : false,
        posture:
          update.posture !== undefined
            ? update.posture
            : idx >= 0
            ? next[idx].posture
            : null,
        heart_rate:
          update.heart_rate !== undefined
            ? update.heart_rate
            : idx >= 0
            ? next[idx].heart_rate
            : null,
        breath_rate:
          update.breath_rate !== undefined
            ? update.breath_rate
            : idx >= 0
            ? next[idx].breath_rate
            : null,
      };
      if (idx >= 0) {
        next[idx] = merged;
      } else {
        next.push(merged);
      }
      return next;
    });
  }, []);

  // Fetch data
  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoadState("loading");
        const families = await getFamilies(token);
        if (!families || families.length === 0) {
          if (!cancelled) {
            setLoadState("empty");
          }
          return;
        }
        const fid = families[0].id as number;
        if (!cancelled) setFamilyId(fid);

        const data = await getToday(token, fid);
        if (!cancelled) {
          const list: RoomData[] = (data.rooms ?? data) ?? [];
          if (list.length === 0) {
            setLoadState("empty");
          } else {
            setRooms(list);
            setLoadState("ok");
          }
        }
      } catch (e: any) {
        if (!cancelled) {
          setErrorMsg(e.message || "加载失败");
          setLoadState("error");
        }
      }
    }
    load();
    return () => { cancelled = true; };
  }, [token]);

  // WebSocket
  useEffect(() => {
    if (!familyId) return;
    const wsUrl = `${WS_BASE}/ws/app?token=${token}&family=${familyId}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (!msg.kind || !msg.payload) return;
        const p = msg.payload;
        if (msg.kind === "posture") {
          mergeUpdate({
            room_name: p.room,
            device_online: true,
            posture: p.posture,
          });
        } else if (msg.kind === "vital") {
          mergeUpdate({
            room_name: p.room,
            device_online: true,
            heart_rate: p.heart_rate,
            breath_rate: p.resp_rate,
          });
        }
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [familyId, token, mergeUpdate]);

  // ---- RENDER: loading ----
  if (loadState === "loading") {
    return (
      <Screen>
        <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Text style={{ color: colors.textMuted, marginTop: space.md, fontSize: font.body }}>
            加载中...
          </Text>
        </View>
      </Screen>
    );
  }

  // ---- RENDER: error ----
  if (loadState === "error") {
    return (
      <Screen>
        <View style={{ flex: 1, justifyContent: "center" }}>
          <Card tone="alert">
            <View style={{ alignItems: "center", padding: space.lg }}>
              <Ionicons name="warning-outline" size={48} color={colors.alert} />
              <Text style={{ fontSize: font.h2, fontWeight: "600", color: colors.text, marginTop: space.md }}>
                加载失败
              </Text>
              <Text style={{ fontSize: font.body, color: colors.textMuted, marginTop: space.xs, textAlign: "center" }}>
                {errorMsg}
              </Text>
            </View>
          </Card>
        </View>
      </Screen>
    );
  }

  // ---- RENDER: empty ----
  if (loadState === "empty") {
    return (
      <Screen>
        <View style={{ flex: 1, justifyContent: "center" }}>
          <Card>
            <View style={{ alignItems: "center", padding: space.lg }}>
              <Ionicons name="wifi-outline" size={48} color={colors.textMuted} />
              <Text style={{ fontSize: font.h2, fontWeight: "600", color: colors.text, marginTop: space.md }}>
                暂无数据
              </Text>
              <Text style={{ fontSize: font.body, color: colors.textMuted, marginTop: space.xs, textAlign: "center" }}>
                您的家庭设备尚未上报数据，请确认设备已连接并正常运行
              </Text>
            </View>
          </Card>
        </View>
      </Screen>
    );
  }

  // ---- RENDER: normal ----
  return (
    <Screen>
      <ScrollView
        contentContainerStyle={{ paddingBottom: space.xxl }}
        showsVerticalScrollIndicator={false}
      >
        <SectionHeader title="今日状态" subtitle="家人各房间实时情况" />

        {rooms.map((room, i) => {
          const cardTone = room.posture === "fall" ? "alert" : "normal";
          return (
            <Card key={room.room_name || i} tone={cardTone}>
              {/* Room header */}
              <View
                style={{
                  flexDirection: "row",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: space.md,
                }}
              >
                <View style={{ flexDirection: "row", alignItems: "center", gap: space.sm }}>
                  <Ionicons name="home-outline" size={20} color={colors.primary} />
                  <Text style={{ fontSize: font.h2, fontWeight: "600", color: colors.text }}>
                    {room.room_name}
                  </Text>
                </View>
                <StatusBadge online={room.device_online} />
              </View>

              {/* Posture */}
              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  marginBottom: room.heart_rate !== null || room.breath_rate !== null ? space.md : 0,
                }}
              >
                <Ionicons
                  name="body-outline"
                  size={18}
                  color={room.posture === "fall" ? colors.alert : colors.primary}
                />
                <Text
                  style={{
                    fontSize: font.body,
                    color: room.posture === "fall" ? colors.alert : colors.text,
                    marginLeft: space.xs,
                    fontWeight: room.posture === "fall" ? "700" : "400",
                  }}
                >
                  当前姿态：{room.posture ? POSTURE_CN[room.posture] ?? room.posture : "未检测"}
                </Text>
              </View>

              {/* Vitals */}
              {(room.heart_rate !== null || room.breath_rate !== null) && (
                <View
                  style={{
                    flexDirection: "row",
                    borderTopWidth: 1,
                    borderTopColor: colors.border,
                    paddingTop: space.md,
                  }}
                >
                  {room.heart_rate !== null && (
                    <VitalStat
                      icon="heart-outline"
                      label="心率"
                      value={`${room.heart_rate}`}
                    />
                  )}
                  {room.breath_rate !== null && (
                    <VitalStat
                      icon="pulse-outline"
                      label="呼吸"
                      value={`${room.breath_rate}`}
                    />
                  )}
                </View>
              )}
            </Card>
          );
        })}

        {/* Logout */}
        <TouchableOpacity
          style={{
            marginTop: space.lg,
            padding: space.md,
            alignItems: "center",
          }}
          onPress={() => {
            wsRef.current?.close();
            onLogout();
          }}
        >
          <Text style={{ color: colors.textMuted, fontSize: font.small }}>
            退出登录
          </Text>
        </TouchableOpacity>
      </ScrollView>
    </Screen>
  );
}
