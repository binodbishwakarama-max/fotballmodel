import { ImageResponse } from "next/og";

export const alt = "BWB Forecast public scoreboard";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "72px",
          background: "linear-gradient(135deg, #07110f 0%, #12231d 100%)",
          color: "#eff7f1",
          fontFamily: "Arial, sans-serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "18px" }}>
          <div
            style={{
              width: "62px",
              height: "62px",
              borderRadius: "18px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: "#b5f36a",
              color: "#14200d",
              fontSize: "34px",
              fontWeight: 900,
            }}
          >
            B
          </div>
          <div style={{ fontSize: "22px", letterSpacing: "5px", color: "#b5f36a" }}>
            BWB FORECAST
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
          <div style={{ fontSize: "68px", fontWeight: 700, letterSpacing: "-3px" }}>
            Public scoreboard
          </div>
          <div style={{ fontSize: "28px", color: "#9bb0a4" }}>
            Football forecasts. Measured in public.
          </div>
        </div>
        <div style={{ width: "100%", height: "2px", background: "#243b31" }} />
      </div>
    ),
    size,
  );
}
