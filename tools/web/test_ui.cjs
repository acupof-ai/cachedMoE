// Run with node --test tools/web/test_ui.cjs. No browser, server, or GPU required.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm');
const page=fs.readFileSync(`${__dirname}/index.html`,'utf8');
const code=page.split('// ---------- response channels')[1].split('// ---------- bubbles')[0];
const Channels=vm.runInNewContext(code.slice(code.indexOf('class ResponseChannels'))+';ResponseChannels');
function parse(text, think, cuts){
  const c=new Channels(think);let at=0;
  for(const n of cuts){c.push(text.slice(at,at+n));at+=n;}
  c.push(text.slice(at));c.push('',{},true);return c;
}
test('reasoning and answer survive every two-chunk boundary, with implicit or echoed opener',()=>{
  for(const open of ['', '<think>']){
    const text=open+'Compare the options.\n</think>\n**Use A.**<｜end▁of▁sentence｜>';
    for(let i=0;i<=text.length;i++){
      const c=parse(text,true,[i]);
      assert.equal(c.reasoning,'Compare the options.\n');assert.equal(c.answer,'\n**Use A.**');
    }
    const c=parse(text,true,Array(text.length).fill(1));assert.equal(c.answer,'\n**Use A.**');
  }
});
test('cancellation or length limit retains unfinished reasoning',()=>{
  const c=parse('Partial reasoning </thi',true,[19,2]);
  assert.equal(c.reasoning,'Partial reasoning </thi');assert.equal(c.answer,'');
});
test('ordinary answers keep literal thinking tags and strip EOS',()=>{
  const c=parse('Example: `<think>text</think>`<｜end▁of▁sentence｜>',false,[2,13,1]);
  assert.equal(c.answer,'Example: `<think>text</think>`');assert.equal(c.reasoning,'');
});
test('a held delimiter does not lose token probability metadata',()=>{
  const c=new Channels(true), first={id:1,p:.1},second={id:2,p:.9};
  c.push('x</th',first);const chunks=c.push('ink>answer',second);
  assert.equal(chunks.length,1);assert.equal(chunks[0].channel,'answer');
  assert.equal(chunks[0].text,'answer');assert.equal(chunks[0].meta,second);
  const d=new Channels(false);d.push('<',first);
  const literal=d.push('example',second);
  assert.equal(literal[0].text,'<');assert.equal(literal[0].meta,first);
});
