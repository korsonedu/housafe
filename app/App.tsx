import { useState } from "react";
import { SafeAreaProvider } from "react-native-safe-area-context";
import LoginScreen from "./src/LoginScreen";
import TodayScreen from "./src/TodayScreen";

export default function App() {
  const [token, setToken] = useState<string | null>(null);
  return (
    <SafeAreaProvider>
      {token ? <TodayScreen token={token} onLogout={() => setToken(null)} /> : <LoginScreen onLogin={setToken} />}
    </SafeAreaProvider>
  );
}
