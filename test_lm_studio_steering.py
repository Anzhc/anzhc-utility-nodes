import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
PACKAGE='anzhc_steering_tests'


def load_nodes():
    package=types.ModuleType(PACKAGE);package.__path__=[str(ROOT)]
    server=types.ModuleType('server')
    server.PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(
        routes=types.SimpleNamespace(get=lambda path:lambda fn:fn)))
    mcp=types.ModuleType(PACKAGE+'.mcp_runtime');mcp.coerce_mcp_integration_config=lambda config:config
    comfy=types.ModuleType('comfy');comfy.__path__=[]
    management=types.ModuleType('comfy.model_management')
    management.processing_interrupted=lambda:False
    management.throw_exception_if_processing_interrupted=lambda:None
    comfy.model_management=management
    modules={PACKAGE:package,'server':server,PACKAGE+'.mcp_runtime':mcp,'comfy':comfy,'comfy.model_management':management}
    stubs=patch.dict(sys.modules,modules);stubs.start()
    loaded=[]
    for name in ('lm_studio_llm_node','lm_studio_pipeline_node','lm_studio_gacha_node'):
        spec=importlib.util.spec_from_file_location(PACKAGE+'.'+name,ROOT/(name+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);loaded.append(module)
    return loaded,stubs


class SteeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.llm,cls.pipeline,cls.gacha),cls.stubs=load_nodes()

    @classmethod
    def tearDownClass(cls):cls.stubs.stop()

    def test_helper_changes_target_and_can_disable(self):
        node=self.llm.AnzhcLMStudioNeuronSteering()
        first=node.configure(True,34,3838,'add',0.75,'residual_norm')[0]
        second=node.configure(True,12,100,'multiply',0,'residual_norm')[0]
        self.assertEqual(json.loads(first)[0]['layer'],34)
        self.assertEqual(json.loads(second),[{'layer':12,'neuron':100,'mode':'multiply','value':0,'unit':'raw'}])
        self.assertEqual(node.configure(False,0,0,'add',0,'raw')[0],'[]')
        chained=node.configure(True,38,17180,'add',0.75,'residual_norm',first)[0]
        self.assertEqual(len(json.loads(chained)),2)
        self.assertEqual(len(json.loads(first)),1)

    def test_models_come_only_from_native_server(self):
        self.llm._MODEL_CACHE.update(models=None,expires_at=0)
        with patch.object(self.llm,'_request_json',return_value={'data':[{'id':'qwen-custom-instance'}]}) as call:
            self.assertEqual(self.llm.fetch_lm_studio_models(force_refresh=True),['qwen-custom-instance'])
            self.assertEqual(call.call_count,1)
            self.assertEqual(call.call_args.args[0],self.llm.LM_STUDIO_MODELS_URL)

    def test_old_node_calls_keep_defaults_and_all_nodes_expose_steering(self):
        with patch.object(self.llm,'get_lm_studio_model_options',return_value=['local']), \
             patch.object(self.pipeline,'get_lm_studio_model_options',return_value=['local']), \
             patch.object(self.gacha,'get_lm_studio_model_options',return_value=['local']):
            for node in (self.llm.AnzhcLMStudioLLM,self.pipeline.AnzhcLMStudioPipeline,self.gacha.AnzhcLMStudioGacha):
                self.assertIn('steering',node.INPUT_TYPES()['optional'])
        with patch.object(self.llm,'call_lm_studio_chat',return_value='ok') as call:
            self.assertEqual(self.llm.AnzhcLMStudioLLM().generate('','Hello','local',0),('ok',))
            self.assertIsNone(call.call_args.kwargs['steering'])

    def test_native_stream_preserves_model_and_request_scoped_targets(self):
        calls=[]
        class Response:
            def raise_for_status(self):pass
            def close(self):pass
            def iter_lines(self,**kwargs):
                yield b'data: {"choices":[{"delta":{"content":"native response"}}]}'
                yield b'data: [DONE]'
        def post(url,**kwargs):calls.append((url,kwargs));return Response()
        with patch('requests.Session',side_effect=lambda:types.SimpleNamespace(post=post,close=lambda:None)), patch.object(self.llm,'_request_json',return_value={'version':1,'backend':'lmstudio-managed'}):
            self.assertEqual(self.llm.call_lm_studio_chat('custom-instance','','Hello',0),'native response')
            for model,layer in (('zwielicht-27b',34),('qwen3.8-queen-27b',12)):
                self.llm.call_lm_studio_chat(model,'','Hello',0,
                    steering=json.dumps([{'layer':layer,'neuron':0,'value':0}]))
            self.llm.call_lm_studio_chat('qwen3.8-queen-27b','','Hello',0,steering='[]')
            self.llm.call_lm_studio_chat('Neuron API: zwielicht-neuron','','Hello',0,steering='[]')
        self.assertTrue(all(url==self.llm.LM_STUDIO_CHAT_URL and kwargs['json']['stream'] for url,kwargs in calls))
        self.assertNotIn('steering',calls[0][1]['json'])
        self.assertEqual([kwargs['json']['model'] for _,kwargs in calls],
                         ['custom-instance','zwielicht-27b','qwen3.8-queen-27b','qwen3.8-queen-27b','zwielicht-27b'])
        self.assertEqual(calls[1][1]['json']['steering'][0]['layer'],34)
        self.assertEqual(calls[2][1]['json']['steering'][0]['layer'],12)
        self.assertEqual(calls[3][1]['json']['steering'],[])

    def test_native_rest_chat_forwards_steering_with_reasoning_or_mcp(self):
        calls=[]
        class Response:
            def raise_for_status(self):pass
            def close(self):pass
            def iter_lines(self,**kwargs):
                yield b'event: chat.end'
                yield b'data: {"result":{"output":[{"type":"message","content":"native result"}]}}'
                yield b''
        def post(url,**kwargs):calls.append((url,kwargs));return Response()
        config='[{"layer":12,"neuron":42,"mode":"multiply","value":1}]'
        with patch('requests.Session',side_effect=lambda:types.SimpleNamespace(post=post,close=lambda:None)), patch.object(self.llm,'_request_json',return_value={'version':1,'backend':'lmstudio-managed'}):
            for kwargs in ({'reasoning_budget':32},{'mcp_config':{}}):
                self.assertEqual(self.llm.call_lm_studio_chat('custom-instance','','Hello',0,
                    max_tokens=64,steering=config,**kwargs),'native result')
        self.assertTrue(all(url==self.llm.LM_STUDIO_NATIVE_CHAT_URL for url,_ in calls))
        self.assertTrue(all(kwargs['json']['model']=='custom-instance' and kwargs['json']['steering']==json.loads(config) for _,kwargs in calls))
        self.assertEqual(calls[0][1]['json']['max_output_tokens'],96)
        self.assertEqual(calls[1][1]['json']['integrations'],[{}])

    def test_missing_native_patch_fails_before_generation(self):
        with patch.object(self.llm,'_request_json',side_effect=RuntimeError('404')), patch.object(self.llm,'_call_lm_studio_openai_chat') as call:
            with self.assertRaisesRegex(RuntimeError,'native steering patch is not active'):
                self.llm.call_lm_studio_chat('custom-instance','','Hello',0,steering='[{"layer":0,"neuron":0,"value":1}]')
            call.assert_not_called()

    def test_pipeline_and_gacha_forward_settings_on_every_call(self):
        config='[{"layer":12,"neuron":42,"mode":"multiply","value":1,"unit":"raw"}]'
        pipeline={'name':'test','user_template':'<text_input> <agent_context>','final_agent':'b','agents':[
            {'id':'a','name':'a','tier':1,'depends_on':[],'system_prompt':'','max_output_tokens':16},
            {'id':'b','name':'b','tier':0,'depends_on':['a'],'system_prompt':'','max_output_tokens':16}]}
        with patch.object(self.pipeline,'call_lm_studio_chat',return_value='2') as call:
            self.assertEqual(self.pipeline.execute_pipeline(pipeline,model='Neuron API: zwielicht-neuron',
                temperature=0,seed=0,text_input='2',steering=config),'2')
            self.assertEqual([c.kwargs['steering'] for c in call.call_args_list],[config,config])
        with patch.object(self.gacha,'call_lm_studio_chat',side_effect=[
                '{"name":"color","ask":"which?","options":["blue"]}',
                '{"options":["blue circle"]}','blue circle']) as call:
            self.gacha._generate_gacha(model='Neuron API: zwielicht-neuron',temperature=0,seed=0,
                decision_steps=1,options_per_step=1,output_token_limit=64,text_input='circle',steering=config)
            self.assertEqual([c.kwargs['steering'] for c in call.call_args_list],[config]*3)


if __name__=='__main__':unittest.main(verbosity=2)
