import type { Metadata } from "next";
import { Inter, Noto_Sans_Devanagari } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/contexts/AuthContext";
import { ChatHistoryProvider } from "@/contexts/ChatHistoryContext";
import QueryProvider from "@/components/providers/QueryProvider";
import { Toaster } from "react-hot-toast";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const devanagari = Noto_Sans_Devanagari({
  subsets: ["devanagari"],
  variable: "--font-devanagari",
  weight: ["400", "600", "700"],
});

export const metadata: Metadata = {
  title: "HealthPolicyLens - Insurance Document Intelligence",
  description:
    "AI-powered platform for analyzing healthcare insurance documents. Upload, extract, summarize and detect risks instantly.",
  keywords: "healthcare insurance, AI analysis, document intelligence, policy summary",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${devanagari.variable} font-sans antialiased bg-[#0a0f1e] text-white`}>
        <QueryProvider>
          <AuthProvider>
            <ChatHistoryProvider>
              {children}
              <Toaster
                position="top-right"
                toastOptions={{
                  style: {
                    background: "#1e2a3a",
                    color: "#e2e8f0",
                    border: "1px solid #2d3748",
                  },
                  success: { iconTheme: { primary: "#10b981", secondary: "#0a0f1e" } },
                  error: { iconTheme: { primary: "#ef4444", secondary: "#0a0f1e" } },
                }}
              />
            </ChatHistoryProvider>
          </AuthProvider>
        </QueryProvider>
      </body>
    </html>
  );
}
