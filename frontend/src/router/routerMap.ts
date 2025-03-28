/**
 * 基础路由
 * @type { *[] }
 */

// const constantRouterMap = [
//   {
//     path: '/',
//     name: 'Screen',
//     redirect: { name: 'ExampleHelloIndex' },
//     children: [
//       {
//         path: '/example',
//         name: 'ExampleHelloIndex',
//         component: () => import('@/views/example/hello/Index.vue')
//       },
//     ]
//   },
// ]

const constantRouterMap = [
  {
    path: '/',
    name: 'ScreenTool',
    redirect: { name: 'ExampleHelloIndex' },
    children: [
      {
        path: '/example',
        name: 'ExampleHelloIndex',
        component: () => import('@/views/screening/MelanomaScreening.vue')
      },
    ]
  },
]

export default constantRouterMap