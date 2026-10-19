"""Protocol mappings: tool histories and terminal events must survive adapters."""
import copy
import io
import json
import queue
import unittest
from types import SimpleNamespace
import api


class ApiTest(unittest.TestCase):
    def test_responses_tool_history(self):
        body = {'input': [{'type': 'function_call', 'call_id': 'c1', 'name': 'shell', 'arguments': '{"cmd":"pwd"}'}, {'type': 'function_call_output', 'call_id': 'c1', 'output': '/tmp'}], 'tools': [{'type': 'function', 'name': 'shell', 'parameters': {'type': 'object'}}]}
        req, tools = api.normalize('/v1/responses', body)
        self.assertTrue(tools)
        self.assertEqual(req['api_messages'][-1], {'role': 'tool', 'tool_call_id': 'c1', 'content': '/tmp'})
        self.assertEqual(req['api_messages'][1]['tool_calls'][0]['id'], 'c1')

    def test_namespace_schema(self):
        req, tools = api.normalize('/v1/responses', {'input': 'hi', 'tools': [{'type': 'namespace', 'name': 'ns', 'tools': [{'type': 'function', 'name': 'run', 'parameters': {'type': 'object'}}]}]})
        self.assertEqual(req['api_messages'][0]['tools'][0]['namespace'], 'ns')
        out = api.result('/v1/responses', {'tool_calls': [{'namespace': 'ns', 'function': {'name': 'run', 'arguments': '{}'}}]}, (1, 2), 'stop', 'r')
        self.assertEqual(out['output'][0]['namespace'], 'ns')
        self.assertEqual(out['output'][0]['name'], 'run')

    def test_anthropic_tools(self):
        req, tools = api.normalize('/v1/messages', {'messages': [{'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'c', 'name': 'shell', 'input': {'cmd': 'pwd'}}]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'c', 'content': '/tmp'}]}]})
        self.assertEqual(req['api_messages'][0]['tool_calls'][0]['function']['arguments'], '{"cmd": "pwd"}')
        self.assertEqual(req['api_messages'][1]['content_blocks'][0]['tool_use_id'], 'c')
        msg = {'content': '', 'tool_calls': [{'type': 'function', 'function': {'name': 'shell', 'arguments': '{"cmd":"pwd"}'}}]}
        out = api.result('/v1/messages', msg, (10, 4), 'stop', 'm')
        self.assertEqual(out['stop_reason'], 'tool_use')
        self.assertEqual(out['content'][0]['input'], {'cmd': 'pwd'})

    def test_reject_unknown_model_and_stateful_responses(self):
        for body in ({'model': 'wrong'}, {'previous_response_id': 'r'}):
            with self.assertRaises(ValueError):
                api.normalize('/v1/responses', body)

    def test_stream_events(self):
        for path in ('/v1/chat/completions', '/v1/messages', '/v1/responses'):
            q = queue.Queue()
            for ev in ({'event': 'start', 'prompt_tokens': 3}, {'event': 'token', 'text': 'Hi'}, {'event': 'done', 'api_message': {'content': 'Hi'}, 'finish': 'stop'}, None):
                q.put(ev)
            h = SimpleNamespace(bridge=SimpleNamespace(args=SimpleNamespace(), submit=lambda *a: SimpleNamespace(out=q)), wfile=io.BytesIO(), send_response=lambda *a: None, send_header=lambda *a: None, end_headers=lambda: None)
            api.handle(h, path, {'messages': [{'role': 'user', 'content': 'Hello'}], 'input': 'Hello', 'stream': True})
            data = h.wfile.getvalue().decode()
            if path.endswith('completions'):
                self.assertIn('data: [DONE]\n\n', data)
            elif path.endswith('messages'):
                events = [line[7:] for line in data.splitlines() if line.startswith('event: ')]
                self.assertEqual(events, ['message_start', 'content_block_start', 'content_block_delta', 'content_block_stop', 'message_delta', 'message_stop'])
            else:
                self.assertIn('event: response.completed', data)
            self.assertTrue(data.endswith('0\r\n\r\n'))


if __name__ == '__main__':
    unittest.main()
