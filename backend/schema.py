"""Wire types. Tensor dimensions omit the batch axis and use C, H, W."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Kind = Literal[
    "grid_input", "feature_input", "conv2d", "dense", "maxpool2d",
    "padding", "flatten", "concat", "residual", "classification", "regression",
    "convnext_stage", "layernorm", "global_avgpool", "embedding",
    "sequence_input", "token_input", "token_embedding", "patch_embed", "position_encoding",
    "grid_sequence_input", "rnn", "gru", "lstm", "conv_rnn", "conv_gru", "conv_lstm",
    "attention", "cross_attention", "feedforward", "transformer_block", "token_pool", "token_head",
    "deepseek_stage", "deepseek_vision", "deepseek_bridge", "deepseek_head", "deepseek_draft",
    "activation", "add", "dot", "gather", "globalavgpool2d", "globalmaxpool2d", "upsample2d",
    "legal_policy", "outcome_transform", "standardize", "checkpoint_module",
]


class Position(BaseModel):
    x: float = 0
    y: float = 0


class NodeData(BaseModel):
    model_config = ConfigDict(extra="allow")
    kind: Kind
    label: str = Field(min_length=1, max_length=100)
    params: dict[str, JsonValue] = Field(default_factory=dict)


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(min_length=1, max_length=100)
    type: str = "layer"
    position: Position = Field(default_factory=Position)
    data: NodeData


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(min_length=1, max_length=250)
    source: str
    target: str
    label: str | None = Field(default=None, max_length=500)
    data: dict[str, JsonValue] = Field(default_factory=dict)


class Graph(BaseModel):
    model_config = ConfigDict(extra="ignore")
    version: Literal[1] = 1
    name: str = Field(default="Untitled architecture", max_length=100)
    template: str | None = Field(default=None, max_length=100)
    mode: Literal["builder", "schematic"] = "builder"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    nodes: list[GraphNode] = Field(max_length=300)
    edges: list[GraphEdge] = Field(max_length=2000)

    @model_validator(mode="before")
    @classmethod
    def recognize_schematic(cls, value):
        if isinstance(value, dict) and "mode" not in value and isinstance(value.get("nodes"), list) and value["nodes"]:
            if all(isinstance(n, dict) and isinstance(n.get("data"), dict) and "output_shape" in n["data"] and "parameter_count" in n["data"] for n in value["nodes"]):
                value = {**value, "mode": "schematic"}
        return value
