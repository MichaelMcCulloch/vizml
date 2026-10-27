/** Parse a declared shape without interpreting symbolic dimensions. */
export function parseDeclaredShape(input: string): (number | string)[] {
  const text = input.trim();
  if (!text) throw new Error("Enter at least one dimension.");
  let values: unknown;
  if (text.startsWith("[")) {
    try { values = JSON.parse(text); }
    catch { throw new Error("The shape array is not valid JSON."); }
  } else {
    values = text.split(/\s*[×,]\s*|\s+[xX]\s+/).map(part => {
      const token = part.trim();
      if (/^\d+$/.test(token)) return Number(token);
      if (/^[+-]?\d+(?:\.\d+)?$/.test(token)) throw new Error("Dimensions must be nonnegative integers or symbolic names.");
      return token;
    });
  }
  if (!Array.isArray(values) || values.length < 1 || values.length > 6 ||
      values.some(value => !(Number.isSafeInteger(value) && value >= 0) &&
        !(typeof value === "string" && value.trim().length > 0 && value.length <= 40)))
    throw new Error("Use 1–6 nonnegative integers or symbolic dimensions.");
  // A leading batch marker is excluded by the schematic analyser.
  if (values.length === 1 && values[0] === "B")
    throw new Error("Add at least one dimension after the batch marker B.");
  return values as (number | string)[];
}

export function formatDeclaredShape(shape: unknown): string {
  if (!Array.isArray(shape)) return "";
  // Keep unusual symbolic strings lossless; the compact form is for the
  // ordinary integer and named dimensions people type by hand.
  if (shape.some(value => typeof value === "string" && (/[×,]|\s[xX]\s/.test(value) || /^[+-]?\d+(?:\.\d+)?$/.test(value) || /^\s|\s$/.test(value))))
    return JSON.stringify(shape);
  return shape.join(" × ");
}
