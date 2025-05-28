import { type Plugin } from 'vite';
import { type PolyfillOptions, nodePolyfills } from 'vite-plugin-node-polyfills';

/**
 * 修复 vite-plugin-node-polyfills 查找不到源文件地址问题
 *
 * https://github.com/davidmyersdev/vite-plugin-node-polyfills/issues/81
 */
const nodePolyfillsFix = (options?: PolyfillOptions): Plugin => {
    return {
        ...nodePolyfills(options),
        resolveId(source: string) {
            const module = /^vite-plugin-node-polyfills\/shims\/(buffer|global|process)$/.exec(
                source,
            );
            if (module) {
                return `node_modules/vite-plugin-node-polyfills/shims/${module[1]}/dist/index.cjs`;
            }
        },
    };
};

export default nodePolyfillsFix;
