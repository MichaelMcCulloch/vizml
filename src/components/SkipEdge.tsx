import { BaseEdge, type EdgeProps } from "@xyflow/react";

export default function SkipEdge({ id, sourceX, sourceY, targetX, targetY, style, label, markerEnd }: EdgeProps) {
  const lift = Math.min(160, Math.max(90, Math.abs(targetX - sourceX) * 0.25));
  const bend = Math.min(90, Math.abs(targetX - sourceX) / 3);
  const path = `M ${sourceX} ${sourceY} C ${sourceX + bend} ${sourceY - lift}, ${targetX - bend} ${targetY - lift}, ${targetX} ${targetY}`;
  return <>
    <BaseEdge id={id} path={path} style={style} markerEnd={markerEnd} />
    {label && <text x={(sourceX + targetX) / 2} y={(sourceY + targetY) / 2 - lift * .75 - 9} textAnchor="middle" fill="#cbaa90" fontSize={10}>{label}</text>}
  </>;
}
