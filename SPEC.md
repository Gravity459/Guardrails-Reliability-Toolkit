# SPEC.md

## 1. JSON Schema

```json
{
  "type": "object",
  "properties": {
    "model": {
      "type": "string",
      "description": "Name or model of the mobile phone"
    },
    "storage": {
      "type": "string",
      "description": "Storage capacity of the phone (e.g., 128GB, 256GB)"
    },
    "camera": {
      "type": "string",
      "description": "Camera specifications (e.g., 50MP dual camera)"
    },
    "battery": {
      "type": "string",
      "description": "Battery capacity (e.g., 5000mAh)"
    },
    "year": {
      "type": "integer",
      "description": "Year the phone was released"
    }
  },
  "required": ["model", "storage", "camera", "battery", "year"]
}
```

---

## 2. Models Used

* **Primary Model:** `gemini-3.5-flash-lite`
* **Fallback Model:** `gemini-3.6-flash`
