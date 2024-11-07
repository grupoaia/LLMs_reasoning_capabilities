from typing import Any, List
from langchain_core.outputs import Generation
from langchain_core.output_parsers.transform import BaseCumulativeTransformOutputParser


class PythonScriptParser(BaseCumulativeTransformOutputParser[Any]):

    def parse_result(self, result: List[Generation], partial: bool = False) -> Any:
        text = result[0].text
        text = text.strip()
        pos0 = 0 if text.find("python") == -1 else text.find("python") + len("python\n")
        parsed_text = text[pos0:]
        if "`" in parsed_text:
            return parsed_text[: parsed_text.find("`")].strip()
        else:
            return parsed_text.strip()

    def parse(self, text: str) -> Any:
        return self.parse_result([Generation(text=text)])
