// The following Demo is adopted from eli thegreenplace.
// https://eli.thegreenplace.net/2011/05/17/the-curiously-recurring-template-pattern-in-c
#include <iostream>
using namespace std;

struct TreeNode
{
    enum Kind {RED, BLUE};

    TreeNode(Kind kind_, TreeNode* left_ = NULL, TreeNode* right_ = NULL)
        : kind(kind_), left(left_), right(right_)
    {}

    Kind kind;
    TreeNode *left, *right;
};

template <typename Derived>
class GenericVisitor
{
public:
    void visit_preorder(TreeNode* node)
    {
        if (node) {
            dispatch_node(node);
            visit_preorder(node->left);
            visit_preorder(node->right);
        }
    }

    void visit_inorder(TreeNode* node)
    {
        if (node) {
            visit_inorder(node->left);
            dispatch_node(node);
            visit_inorder(node->right);
        }
    }

    void visit_postorder(TreeNode* node)
    {
        if (node) {
            visit_postorder(node->left);
            visit_postorder(node->right);
            dispatch_node(node);
        }
    }

    void handle_RED(TreeNode* node)
    {
        cerr << "Generic handle RED\n";
    }

    void handle_BLUE(TreeNode* node)
    {
        cerr << "Generic handle BLUE\n";
    }

private:
    // Convenience method for CRTP
    //
    Derived& derived()
    {
        return *static_cast<Derived*>(this);
    }

    void dispatch_node(TreeNode* node)
    {
        switch (node->kind) {
            case TreeNode::RED:
                derived().handle_RED(node);
                break;
            case TreeNode::BLUE:
                derived().handle_BLUE(node);
                break;
            default:
                assert(0);
        }
    }
};

class SpecialVisitor : public GenericVisitor<SpecialVisitor>
{
public:
    void handle_RED(TreeNode* node)
    {
        cerr << "RED is special\n";
    }
};

int main()
{
  struct TreeNode *c1 = new TreeNode(TreeNode::Kind::RED);
  struct TreeNode *c2 = new TreeNode(TreeNode::Kind::BLUE);
  struct TreeNode *r0 = new TreeNode(TreeNode::Kind::RED, c1, c2);

  SpecialVisitor sv;
  sv.visit_inorder(r0);

  return 0;
}
