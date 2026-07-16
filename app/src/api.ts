export const API_BASE = "http://192.168.1.100:8000"; // 改成你本机局域网 IP
export const WS_BASE = "ws://192.168.1.100:8000";

export async function login(username: string, password: string) {
  const r = await fetch(`${API_BASE}/api/auth/login`, {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ username, password }),
  });
  if (!r.ok) throw new Error("登录失败");
  return (await r.json()).access as string;
}

export async function getFamilies(token: string) {
  const r = await fetch(`${API_BASE}/api/families`, { headers:{ Authorization:`Bearer ${token}` }});
  if (!r.ok) throw new Error("加载家庭失败");
  return r.json();
}

export async function getToday(token: string, familyId: number) {
  const r = await fetch(`${API_BASE}/api/families/${familyId}/today`, { headers:{ Authorization:`Bearer ${token}` }});
  if (!r.ok) throw new Error("加载状态失败");
  return r.json();
}
