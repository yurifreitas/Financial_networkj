import React from "react";

export default function SignalBadge({ value }: { value: -1 | 0 | 1 }) {
  const label = value === 1 ? "COMPRAR" : value === -1 ? "VENDER" : "NEUTRO";
  const cls =
    value === 1 ? "tag ok" : value === -1 ? "tag bad" : "tag warn";
  return <span className={cls}>{label}</span>;
}
