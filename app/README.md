# 家安 App (Expo)

养老监护 App -- 登录后查看家人各房间实时状态。

## 前置条件

- Node.js 20+
- Expo Go（手机安装）或模拟器

## 配置 API 地址

编辑 `src/api.ts`，将 `API_BASE` 和 `WS_BASE` 改为你本机局域网 IP（真机连不上 localhost）：

```ts
export const API_BASE = "http://192.168.x.x:8000";
export const WS_BASE = "ws://192.168.x.x:8000";
```

## 运行

```bash
npx expo start
```

然后用 Expo Go 扫码，或按 `i`（iOS 模拟器）/ `a`（Android 模拟器）。
