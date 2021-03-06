# -*- coding: utf-8 -*-
import collections

def _get_network_type():
    """Returns the type of the outputs of a value distribution network.

    Returns:
        net_type: _network_type object defining the outputs of the network.
    """
    return collections.namedtuple('c51_network',
                                    ['q_values', 'logits', 'probabilities'])

print(_get_network_type()('a', 'b','c'))


# from http://www.zlovezl.cn/articles/collections-in-python/
"""
比如我们用户拥有一个这样的数据结构，每一个对象是拥有三个元素的tuple。
使用namedtuple方法就可以方便的通过tuple来生成可读性更高也更好用的数据结构。
"""
websites = [
    ('Sohu', 'http://www.google.com/', u'张朝阳'),
    ('Sina', 'http://www.sina.com.cn/', u'王志东'),
    ('163', 'http://www.163.com/', u'丁磊')
]

Website = collections.namedtuple('Website', ['name', 'url', 'founder'])

for website in websites:
    website = Website._make(website)
    print(website)


# Result:
# Website(name='Sohu', url='http://www.google.com/', founder=u'\u5f20\u671d\u9633')
# Website(name='Sina', url='http://www.sina.com.cn/', founder=u'\u738b\u5fd7\u4e1c')
# Website(name='163', url='http://www.163.com/', founder=u'\u4e01\u78ca')